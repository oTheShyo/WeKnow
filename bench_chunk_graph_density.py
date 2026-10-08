"""How dense should the chunk-graph projection be?

The query expansion costs `sum(out-degree of frontier)` rows to scan, and `max_chunks_per_entity`
sets that out-degree. At the default 64 the NFCorpus projection has 3.0M edges for 3,633 chunks —
the heaviest chunk has 2,852 of a possible 3,632 — so the expansion reaches ~70% of the corpus and
each query scans ~1M edge rows. This rebuilds the projection at several caps and measures edges,
build time and query latency, so the default is chosen from data rather than taste.

Only `graph_chunk_relation` is rebuilt; the entity graph is reused.
"""

from __future__ import annotations

import time
from pathlib import Path

import ir_datasets
from platformdirs import user_data_dir
from sqlmodel import Session, delete as sa_model_delete

from weknow._config import RAGLiteConfig
from weknow._database import GraphChunkRelation, create_database_engine
from weknow._graph import CooccurrenceExtractor, GraphConfig, build_chunk_relations, graph_search

DB_NAME = "graph_beir-nfcorpus.db"
CAPS = [4, 8, 16, 32, 64]
QUERIES = 5

db_path = Path(user_data_dir("weknow", ensure_exists=True)) / DB_NAME
dataset = ir_datasets.load("beir/nfcorpus/test")
queries = [query.text for query in dataset.queries_iter()][:QUERIES]

print(f"{'cap':>4s} | {'edges':>10s} | {'build':>7s} | {'query mean':>10s} | {'chunks reached':>14s}")
print("-" * 60)
for cap in CAPS:
    config = RAGLiteConfig(
        llm="gpt-4o-mini",
        embedder="text-embedding-3-small",
        reranker=None,
        self_query=False,
        db_url=f"duckdb:///{db_path.as_posix()}",
        graph=GraphConfig(
            enabled=True,
            extractor=CooccurrenceExtractor(),
            build_chunk_graph=True,
            max_chunks_per_entity=cap,
        ),
    )
    engine = create_database_engine(config)
    with Session(engine) as session:
        session.exec(sa_model_delete(GraphChunkRelation))  # type: ignore[arg-type]
        session.commit()
    start = time.perf_counter()
    edges = build_chunk_relations(config=config)
    build_seconds = time.perf_counter() - start

    timings: list[float] = []
    reached: list[int] = []
    for query in queries:
        start = time.perf_counter()
        chunk_ids, _ = graph_search(query, num_results=200, config=config)
        timings.append(time.perf_counter() - start)
        reached.append(len(chunk_ids))
    print(
        f"{cap:4d} | {edges:10,d} | {build_seconds:6.2f}s |"
        f" {sum(timings) / len(timings):9.3f}s | {sum(reached) / len(reached):14.0f}"
    )
