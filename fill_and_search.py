"""Fill a 10k-vector DuckDB database with native SQL, then measure search latency.

Bypasses insert_documents (whose ORM flush path is not the target of this
benchmark) and measures the real search engine: DuckDB VSS HNSW, FTS/BM25,
RRF fusion, chunk spans, and FlashRank cross-encoder rerank.
"""

import json
import statistics
import time
from pathlib import Path

import duckdb
import numpy as np

from bench_resume import BASE_KWARGS, EMBED_DIM, NUM_QUERIES, pct
from raglite import RAGLiteConfig
from raglite._database import create_database_engine

DB_FILE = "bench_data.db"
NUM_CHUNKS = 1000
CHUNKLETS_PER_CHUNK = 10  # -> 10k chunk embeddings
MARKDOWN = Path(__file__).parent.joinpath("bench_markdown.txt").read_text(encoding="utf-8")


def fill_database(config: RAGLiteConfig) -> None:
    """Create tables + indexes via raglite, bulk-load rows via native DuckDB SQL."""
    t0 = time.perf_counter()
    create_database_engine(config).dispose()  # Create tables + indexes, then release the pool.
    engine_setup_s = time.perf_counter() - t0
    print(f"create_database_engine: {engine_setup_s:.2f}s")

    con = duckdb.connect(DB_FILE)
    con.execute("LOAD vss")
    con.execute("LOAD fts")
    con.execute("SET hnsw_enable_experimental_persistence = true")
    existing = con.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    if existing >= NUM_CHUNKS:
        print(f"database already filled ({existing} chunks), skipping bulk insert")
    else:
        con.execute("INSERT INTO document VALUES ('synth-doc', 'synthetic-report.md', NULL, '{}')")

    if existing >= NUM_CHUNKS:
        print(f"database already filled ({existing} chunks), skipping bulk insert")
        chunk_insert_s = embedding_insert_s = float("nan")
    else:
        con.execute("INSERT INTO document VALUES ('synth-doc', 'synthetic-report.md', NULL, '{}')")

        # 1000 chunks of ~1800 chars each, cycled from the real markdown.
        window = len(MARKDOWN) - 1800
        chunk_rows = [
            (f"synth-{i:05d}", "synth-doc", i, "", MARKDOWN[(i * 1800) % window : (i * 1800) % window + 1800], "{}")
            for i in range(NUM_CHUNKS)
        ]
        num_chunks = len(chunk_rows)

        t0 = time.perf_counter()
        con.executemany("INSERT INTO chunk VALUES (?, ?, ?, ?, ?, ?)", chunk_rows)
        chunk_insert_s = time.perf_counter() - t0

        # 10 chunklet embeddings per chunk -> 10k rows of FLOAT[1536], via Parquet bulk import.
        import pyarrow as pa
        import pyarrow.parquet as pq

        rng = np.random.default_rng(42)
        t0 = time.perf_counter()
        embeddings = rng.standard_normal((num_chunks * CHUNKLETS_PER_CHUNK, EMBED_DIM)).astype(
            np.float32
        )
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        chunk_ids = np.repeat([row[0] for row in chunk_rows], CHUNKLETS_PER_CHUNK)
        flat = pa.array(embeddings.reshape(-1))
        vectors = pa.FixedSizeListArray.from_arrays(flat, EMBED_DIM)
        pq.write_table(
            pa.table({"chunk_id": pa.array(chunk_ids), "embedding": vectors}),
            "bench_embeddings.parquet",
        )
        con.execute(
            "INSERT INTO chunk_embedding (chunk_id, embedding) "
            "SELECT chunk_id, embedding FROM read_parquet('bench_embeddings.parquet')"
        )
        embedding_insert_s = time.perf_counter() - t0
    num_chunks = con.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    num_embeddings = con.execute("SELECT COUNT(*) FROM chunk_embedding").fetchone()[0]
    print(
        f"chunks={num_chunks} (insert {chunk_insert_s:.2f}s), "
        f"embeddings={num_embeddings} (insert {embedding_insert_s:.2f}s)"
    )

    # Rebuild the HNSW index so it includes the bulk-loaded rows.
    t0 = time.perf_counter()
    con.execute("DROP INDEX IF EXISTS vector_search_chunk_index")
    con.execute(
        "CREATE INDEX vector_search_chunk_index ON chunk_embedding USING HNSW (embedding) "
        "WITH (metric = 'cosine')"
    )
    hnsw_build_s = time.perf_counter() - t0
    # Build the FTS (BM25) index over chunk bodies.
    t0 = time.perf_counter()
    con.execute("PRAGMA create_fts_index('chunk', 'id', 'body', overwrite = 1)")
    fts_build_s = time.perf_counter() - t0
    con.execute("CHECKPOINT")
    con.close()
    print(f"HNSW index build ({num_embeddings} x {EMBED_DIM}): {hnsw_build_s:.2f}s")
    print(f"FTS index build ({num_chunks} chunks): {fts_build_s:.2f}s")
    return {
        "num_chunks": num_chunks,
        "num_chunk_embeddings": num_embeddings,
        "bulk_insert_chunks_s": round(chunk_insert_s, 2),
        "bulk_insert_embeddings_s": round(embedding_insert_s, 2),
        "hnsw_index_build_s": round(hnsw_build_s, 2),
        "fts_index_build_s": round(fts_build_s, 2),
    }


def bench_search(config: RAGLiteConfig) -> dict:
    from sqlalchemy import create_engine
    from sqlalchemy.pool import SingletonThreadPool
    from sqlmodel import Session, text

    import raglite._database as db_module
    import raglite._search as search_module

    # DuckDB VSS loads the HNSW index into memory per connection, so pin all raglite
    # sessions to a single pooled connection to measure steady-state query latency.
    engine = create_engine(f"duckdb:///{DB_FILE}", poolclass=SingletonThreadPool)
    with Session(engine) as session:
        session.execute(text("INSTALL fts; LOAD fts;"))
        session.execute(text("INSTALL vss; LOAD vss;"))
        session.execute(text("SET hnsw_enable_experimental_persistence = true"))
        session.execute(text("SET hnsw_ef_search = 160"))
        session.commit()
    engine.connect()  # Warm the single pooled connection (loads the HNSW index once).
    db_module.create_database_engine = lambda config=None: engine
    search_module.create_database_engine = db_module.create_database_engine

    from raglite._search import (
        keyword_search,
        reciprocal_rank_fusion,
        retrieve_chunk_spans,
        rerank_chunks,
        vector_search,
    )

    rng = np.random.default_rng(7)
    query_vectors = [
        (lambda e: e / np.linalg.norm(e))(rng.standard_normal(EMBED_DIM).astype(np.float32))
        for _ in range(NUM_QUERIES)
    ]

    vector_times = []
    for i in range(NUM_QUERIES + 5):
        t0 = time.perf_counter()
        vs_ids, _ = vector_search(query_vectors[i % NUM_QUERIES], num_results=20, config=config)
        vector_times.append((time.perf_counter() - t0) * 1e3)
    vector_times = vector_times[5:]

    keyword_times = []
    queries = [
        "space time interval light speed",
        "moving bodies electrodynamics",
        "relativity principle simultaneity",
        "energy mass equivalence",
        "gravitational field equations",
    ]
    for i in range(NUM_QUERIES):
        t0 = time.perf_counter()
        ks_ids, _ = keyword_search(queries[i % 5], num_results=20, config=config)
        keyword_times.append((time.perf_counter() - t0) * 1e3)

    rrf_times = []
    for _ in range(200):
        t0 = time.perf_counter()
        reciprocal_rank_fusion([vs_ids[:40], ks_ids[:40]], weights=[0.75, 0.25])
        rrf_times.append((time.perf_counter() - t0) * 1e3)

    span_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        spans = retrieve_chunk_spans(vs_ids[:8], config=config)
        span_times.append((time.perf_counter() - t0) * 1e3)

    rerank_times, reranker_ok = [], True
    try:
        from dataclasses import replace

        from platformdirs import user_data_dir

        from rerankers import Reranker

        rerank_config = replace(
            config,
            reranker=Reranker(
                "ms-marco-MiniLM-L-12-v2",
                model_type="flashrank",
                verbose=0,
                cache_dir=user_data_dir("raglite", ensure_exists=True),
            ),
        )
        for _ in range(20):
            t0 = time.perf_counter()
            reranked = rerank_chunks("space time interval light speed", vs_ids[:20], config=rerank_config)
            rerank_times.append((time.perf_counter() - t0) * 1e3)
    except Exception as error:  # noqa: BLE001
        reranker_ok = False
        print(f"reranker skipped: {error}")

    return {
        "vector_search_top20_ms": {
            "p50": round(pct(vector_times, 50), 1),
            "p95": round(pct(vector_times, 95), 1),
            "mean": round(statistics.mean(vector_times), 1),
        },
        "keyword_search_top20_ms": {
            "p50": round(pct(keyword_times, 50), 1),
            "p95": round(pct(keyword_times, 95), 1),
        },
        "rrf_fusion_ms": {"p50": round(pct(rrf_times, 50), 2)},
        "chunk_spans_8_results_ms": {"p50": round(pct(span_times, 50), 1)},
        "rerank_20_chunks_ms": {
            "p50": round(pct(rerank_times, 50), 1),
            "p95": round(pct(rerank_times, 95), 1),
        }
        if reranker_ok
        else None,
        "num_span_chunks": len(spans),
        "num_reranked": len(reranked) if reranker_ok else 0,
    }


def main() -> None:
    import os

    os.chdir(Path(__file__).parent)
    config = RAGLiteConfig(db_url=f"duckdb:///{DB_FILE}", reranker=None, **BASE_KWARGS)
    fill_stats = fill_database(config)
    search_stats = bench_search(config)
    output = {"fill": fill_stats, "search": search_stats}
    Path(__file__).parent.joinpath("bench_results.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
