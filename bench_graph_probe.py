"""Stage the graph build so its cost can be attributed, and time it end to end.

`bench_beir_graph.py` ran for 12+ minutes on NFCorpus without finishing the graph build, which
made the ablation impractical to iterate on. This probe does the same work with each stage
timed and printed the moment it finishes - extraction, `_write_graph`,
`recompute_relation_weights`, `build_chunk_relations` - so the cost is attributable even when a
stage does not return.

It indexes the corpus itself if the database has no chunks, so it can be run standalone:

    .venv/Scripts/python.exe bench_graph_probe.py                       # nfcorpus, defaults
    .venv/Scripts/python.exe bench_graph_probe.py beir/scifact/test graph_beir-scifact.db 2 8 32 16

Trailing arguments are `min_occurrences max_entities_per_chunk max_relations_per_chunk
max_chunks_per_entity`.
"""

import faulthandler
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import ir_datasets
from platformdirs import user_data_dir
from sqlalchemy import delete as sa_delete
from sqlalchemy import func
from sqlmodel import Session, col, select

from bench_beir import BASE_KWARGS, BM25Evaluator
from weknow._config import RAGLiteConfig
from weknow._database import Chunk, create_database_engine
from weknow._graph import (
    CooccurrenceExtractor,
    GraphChunk,
    GraphChunkRelation,
    GraphConfig,
    GraphEntity,
    GraphEntityChunk,
    GraphRelation,
    _write_graph,
    build_chunk_relations,
    graph_stats,
    recompute_relation_weights,
)

DATASET_ID = sys.argv[1] if len(sys.argv) > 1 else "beir/nfcorpus/test"
DB_NAME = sys.argv[2] if len(sys.argv) > 2 else "graph_beir-nfcorpus.db"
MIN_OCCURRENCES = int(sys.argv[3]) if len(sys.argv) > 3 else 2
MAX_ENTITIES_PER_CHUNK = int(sys.argv[4]) if len(sys.argv) > 4 else 16
MAX_RELATIONS_PER_CHUNK = int(sys.argv[5]) if len(sys.argv) > 5 else 128
MAX_CHUNKS_PER_ENTITY = int(sys.argv[6]) if len(sys.argv) > 6 else 64


def report(label: str, seconds: float, **extra: Any) -> None:
    tail = " ".join(f"{key}={value}" for key, value in extra.items())
    print(f"{label:<30s} {seconds:9.2f}s   {tail}", flush=True)


def main() -> None:
    # STACK_EVERY=<seconds> dumps every thread's stack on that interval, so a stage that does not
    # return still reports where it is.
    stack_every = float(os.environ.get("STACK_EVERY", "0") or 0)
    if stack_every:
        faulthandler.dump_traceback_later(stack_every, repeat=True)
    db_path = Path(user_data_dir("weknow", ensure_exists=True)) / DB_NAME
    dataset = ir_datasets.load(DATASET_ID)
    extractor = CooccurrenceExtractor(
        min_occurrences=MIN_OCCURRENCES,
        max_entities_per_chunk=MAX_ENTITIES_PER_CHUNK,
        max_relations_per_chunk=MAX_RELATIONS_PER_CHUNK,
    )
    config = RAGLiteConfig(
        **BASE_KWARGS,
        db_url=f"duckdb:///{db_path.as_posix()}",
        graph=GraphConfig(
            enabled=True,
            extractor=extractor,
            build_chunk_graph=True,
            max_chunks_per_entity=MAX_CHUNKS_PER_ENTITY,
        ),
    )
    print(f"dataset   : {DATASET_ID} ({dataset.docs_count()} docs)")
    print(f"database  : {db_path} (exists={db_path.exists()})")
    print(
        f"extractor : min_occurrences={MIN_OCCURRENCES} "
        f"max_entities_per_chunk={MAX_ENTITIES_PER_CHUNK} "
        f"max_relations_per_chunk={MAX_RELATIONS_PER_CHUNK} "
        f"max_chunks_per_entity={MAX_CHUNKS_PER_ENTITY}"
    )
    print(flush=True)

    # Stage 0: index the corpus if this is a fresh database, and clear any partial graph from an
    # earlier aborted run.
    engine = create_database_engine(config)
    with Session(engine) as session:
        indexed_chunks = session.exec(select(func.count(col(Chunk.id)))).one()
    if not indexed_chunks:
        # Reuse the benchmark's bulk loader rather than the ORM path, for the same reason as in
        # `bench_beir.py`: the ORM flush is orders of magnitude slower and is not what is measured.
        indexer = BM25Evaluator(dataset)
        indexer.config = config
        indexer.db_filepath = db_path
        start = time.perf_counter()
        indexer.insert_documents()
        report("0. index corpus", time.perf_counter() - start, chunks=dataset.docs_count())
    else:
        print(f"corpus already indexed: {indexed_chunks} chunks", flush=True)
    with Session(engine) as session:
        for table in (GraphChunkRelation, GraphEntityChunk, GraphRelation, GraphEntity, GraphChunk):
            session.exec(sa_delete(table))  # type: ignore[arg-type]
        session.commit()
        rows: list[tuple[str, str, str, str]] = [
            (chunk_id, document_id, headings, body)
            for chunk_id, document_id, headings, body in session.exec(
                select(Chunk.id, Chunk.document_id, Chunk.headings, Chunk.body)
            ).all()
        ]

    # Stage 1: extraction only, in the same shape `build_graph` uses.
    texts = [f"{headings}\n{body}" if headings else body for _, _, headings, body in rows]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        extractions = list(pool.map(extractor.extract, texts))
    entity_mentions = sum(len(extraction.entities) for extraction in extractions)
    relation_mentions = sum(len(extraction.relations) for extraction in extractions)
    report(
        "1. extraction",
        time.perf_counter() - start,
        chunks=len(rows),
        entities=entity_mentions,
        relations=relation_mentions,
        distinct_titles=len(
            {
                entity.title.casefold()
                for extraction in extractions
                for entity in extraction.entities
            }
        ),
    )

    # COUNT_ONLY=1 stops after extraction, which is enough to size the graph without paying for the
    # write, and makes parameter sweeps cheap.
    if os.environ.get("COUNT_ONLY"):
        print("\nCOUNT_ONLY: stopping before the write stages", flush=True)
        return

    # Stage 2: write the graph tables.
    start = time.perf_counter()
    _write_graph([(row[0], row[1]) for row in rows], extractions, config=config)
    report("2. _write_graph", time.perf_counter() - start)

    # Stage 3: score the edges.
    start = time.perf_counter()
    scored = recompute_relation_weights(config=config)
    report("3. recompute_relation_weights", time.perf_counter() - start, scored=scored)

    # Stage 4: project the entity graph onto the chunks.
    start = time.perf_counter()
    edges = build_chunk_relations(config=config)
    report("4. build_chunk_relations", time.perf_counter() - start, edges=edges)

    print(flush=True)
    print("graph size")
    for key, value in graph_stats(config=config).items():
        print(f"  {key:<26s} : {value}")


if __name__ == "__main__":
    main()
