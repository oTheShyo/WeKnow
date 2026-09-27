"""Stage-by-stage timing of a single-document insertion to locate the bottleneck."""

import sys
import time
from pathlib import Path

import raglite._insert as insert_module
from bench_resume import BASE_KWARGS, _cached_split_sentences, mock_embeddings  # noqa: E402
import bench_resume  # noqa: E402
from raglite import Document, RAGLiteConfig  # noqa: E402
from raglite._database import create_database_engine  # noqa: E402
from raglite._split_sentences import split_sentences as real_split_sentences  # noqa: E402
from sqlmodel import Session  # noqa: E402

markdown = Path(__file__).parent.joinpath("bench_markdown.txt").read_text(encoding="utf-8")
bench_resume._PRECOMPUTED_SENTENCES = real_split_sentences(markdown, max_len=2048)

config = RAGLiteConfig(db_url="duckdb:///bench_probe2.db", reranker=None, **BASE_KWARGS)
insert_module.embed_strings = mock_embeddings
insert_module.embed_strings_without_late_chunking = mock_embeddings
insert_module.split_sentences = _cached_split_sentences

doc = Document.from_text(markdown, id="probe2-doc")

t0 = time.perf_counter()
doc_record, chunk_records, chunk_embedding_records_list = insert_module._create_chunk_records(
    doc, config
)
t1 = time.perf_counter()
num_embeddings = sum(len(group) for group in chunk_embedding_records_list)
print(f"_create_chunk_records: {t1 - t0:.2f}s (chunks={len(chunk_records)}, embeddings={num_embeddings})")

t1 = time.perf_counter()
engine = create_database_engine(config)
t2 = time.perf_counter()
print(f"create_database_engine (incl. CREATE INDEX): {t2 - t1:.2f}s")

with Session(engine, expire_on_commit=False) as session:
    session.add(doc_record)
    ta = time.perf_counter()
    session.add_all(chunk_records)
    session.flush()
    tb = time.perf_counter()
    print(f"add+flush {len(chunk_records)} chunk rows: {tb - ta:.2f}s")
    flushed = 0
    for group in chunk_embedding_records_list:
        session.add_all(group)
        flushed += len(group)
        if flushed >= 128:
            tf = time.perf_counter()
            session.flush()
            print(f"flush {flushed} embedding rows: {time.perf_counter() - tf:.2f}s")
            flushed = 0
    if flushed:
        tf = time.perf_counter()
        session.flush()
        print(f"flush {flushed} embedding rows: {time.perf_counter() - tf:.2f}s")
    tc = time.perf_counter()
    session.commit()
    print(f"commit (incl. FTS/HNSW maintenance): {time.perf_counter() - tc:.2f}s")
print("done")
