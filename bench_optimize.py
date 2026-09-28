"""Diagnose retrieval latency: is the ANN index used, and where does the time actually go?

Answers four questions on the existing 10k-vector benchmark database:
  1. Does DuckDB VSS actually use the HNSW index for RAGLite's query shape, or does it
     silently fall back to a brute-force scan?
  2. How much of the online latency is the cross-encoder reranker, and how does it scale
     with the number of candidates?
  3. What does a cheaper configuration (fewer candidates) buy end to end?
  4. Which (index metric, distance operator) combinations are index-served at all?

Embeddings are random unit vectors and the query vectors are random too, so *ranking quality*
is meaningless here -- only latency and query plans are being measured.

Test machine: Raptor Lake 24 logical cores, DuckDB 1.5.5, Windows. Latency is wall-clock and
load sensitive: report it with the machine context, never as a bare number.
"""

import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

EMBED_DIM = 1536
NUM_QUERIES = 40
DB_FILE = "bench_data.db"
RERANK_QUERY = "space time interval light speed"


def pct(values: list[float], p: float) -> float:
    """Percentile of a list of floats."""
    return float(np.percentile(values, p))


def make_engine() -> Any:
    """Open the benchmark database on a single pinned connection with VSS/FTS loaded."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import SingletonThreadPool
    from sqlmodel import Session, text

    engine = create_engine(f"duckdb:///{DB_FILE}", poolclass=SingletonThreadPool)
    with Session(engine) as session:
        session.execute(text("INSTALL fts; LOAD fts;"))
        session.execute(text("INSTALL vss; LOAD vss;"))
        session.execute(text("SET hnsw_enable_experimental_persistence = true"))
        session.execute(text("SET hnsw_ef_search = 160"))
        session.commit()
    engine.connect()  # Warm the pooled connection (loads the HNSW index into memory).
    return engine


def install_engine(engine: Any) -> None:
    """Point raglite at the pinned engine, register the embedding dimension, stub the embedder.

    `create_database_engine` has a side effect we must not lose: it calls
    `ChunkEmbedding.set_embedding_dim(...)`, which sets the `FLOAT[n]` width used when the
    distance operator casts the query parameter. Monkeypatching the engine without also
    registering the dimension leaves `dim = -1`, and the generated SQL becomes
    `array_cosine_distance(embedding, $1::FLOAT[-1])`, which DuckDB rejects with a parse error.

    The query embedder is stubbed with random unit vectors so the whole benchmark, including the
    string-query pipeline, runs offline with no credentials. Ranking quality is meaningless here;
    only latency is measured.
    """
    import raglite._database as db_module
    import raglite._search as search_module
    from raglite._database import ChunkEmbedding

    ChunkEmbedding.set_embedding_dim(EMBED_DIM)
    db_module.create_database_engine = lambda config=None: engine
    search_module.create_database_engine = db_module.create_database_engine

    rng = np.random.default_rng(1234)
    cache: dict[str, np.ndarray] = {}

    def fake_embed_query(query: str, config: Any) -> np.ndarray:
        if query not in cache:
            vec = rng.standard_normal(EMBED_DIM).astype(np.float32)
            cache[query] = vec / np.linalg.norm(vec)
        return cache[query]

    search_module._embed_query_cached = fake_embed_query


def _explain(cursor: Any, sql: str, params: list[Any]) -> tuple[str, bool]:
    """Return the physical plan text and whether it mentions HNSW."""
    cursor.execute("EXPLAIN " + sql, params)
    rows = cursor.fetchall()
    plan = "\n".join(str(value) for row in rows for value in row if value)
    return plan, "hnsw" in plan.lower()


def bench_plan(engine: Any) -> dict[str, Any]:
    """Inspect the physical query plan for both distance-operator shapes."""
    conn = engine.raw_connection()
    cursor = conn.cursor()
    vec = [0.0] * EMBED_DIM
    out: dict[str, Any] = {}
    for label, op in (
        ("raglite_default_cosine", "array_cosine_distance"),
        ("l2sq_index_compatible", "array_distance"),
    ):
        sql = (
            f"SELECT chunk_id, {op}(embedding, CAST(? AS FLOAT[{EMBED_DIM}])) AS dist "
            "FROM chunk_embedding ORDER BY dist LIMIT 200;"
        )
        plan, uses_hnsw = _explain(cursor, sql, [vec])
        out[label] = {"operator": op, "hnsw_used": uses_hnsw, "plan_head": plan[:600]}
    cursor.close()
    return out


def bench_vector(engine: Any, config: Any, query_vectors: list[np.ndarray]) -> dict[str, Any]:
    """Vector search latency: raglite's wrapper vs raw SQL, plus the brute-force floor."""
    from raglite._search import vector_search

    conn = engine.raw_connection()
    cursor = conn.cursor()

    def timeit(fn: Any, runs: int, warmup: int = 5) -> list[float]:
        times = []
        for i in range(runs + warmup):
            t0 = time.perf_counter()
            fn(i)
            times.append((time.perf_counter() - t0) * 1e3)
        return times[warmup:]

    out: dict[str, Any] = {}

    def raglite_call(i: int) -> None:
        vector_search(query_vectors[i % len(query_vectors)], num_results=20, config=config)

    t = timeit(raglite_call, NUM_QUERIES)
    out["raglite_vector_search_top20"] = {"p50": round(pct(t, 50), 1), "p95": round(pct(t, 95), 1)}

    def cosine_call(i: int) -> None:
        cursor.execute(
            "SELECT chunk_id FROM chunk_embedding ORDER BY "
            f"array_cosine_distance(embedding, CAST(? AS FLOAT[{EMBED_DIM}])) LIMIT 200;",
            [query_vectors[i % len(query_vectors)].tolist()],
        )
        cursor.fetchall()

    t = timeit(cosine_call, NUM_QUERIES)
    out["raw_cosine_distance_top200"] = {"p50": round(pct(t, 50), 1), "p95": round(pct(t, 95), 1)}

    def l2_call(i: int) -> None:
        cursor.execute(
            "SELECT chunk_id FROM chunk_embedding ORDER BY "
            f"array_distance(embedding, CAST(? AS FLOAT[{EMBED_DIM}])) LIMIT 200;",
            [query_vectors[i % len(query_vectors)].tolist()],
        )
        cursor.fetchall()

    t = timeit(l2_call, NUM_QUERIES)
    out["raw_l2_distance_top200"] = {"p50": round(pct(t, 50), 1), "p95": round(pct(t, 95), 1)}

    # Brute-force floor: aggregate over every row, no ORDER BY / LIMIT to push down.
    def scan_call(i: int) -> None:
        cursor.execute(
            "SELECT min(array_cosine_distance(embedding, "
            f"CAST(? AS FLOAT[{EMBED_DIM}]))) FROM chunk_embedding;",
            [query_vectors[i % len(query_vectors)].tolist()],
        )
        cursor.fetchone()

    t = timeit(scan_call, 15)
    out["full_scan_floor_min_cosine_distance"] = {"p50": round(pct(t, 50), 1)}
    cursor.close()
    return out


def bench_rerank(config: Any, num_chunks: int) -> dict[str, Any]:
    """Cross-encoder rerank latency for a candidate count, with warmup discarded."""
    from platformdirs import user_data_dir
    from rerankers import Reranker
    from sqlmodel import Session, select

    from raglite._database import ChunkEmbedding
    from raglite._search import rerank_chunks

    # Point flashrank at the already-extracted local cache so the benchmark never needs the network.
    rerank_config = replace(
        config,
        reranker=Reranker(
            "ms-marco-MiniLM-L-12-v2",
            model_type="flashrank",
            verbose=0,
            cache_dir=user_data_dir("raglite", ensure_exists=True),
        ),
    )
    with Session(create_bench_engine()) as session:
        rows = session.exec(select(ChunkEmbedding.chunk_id).limit(num_chunks)).all()
    chunk_ids = [row for row in rows][:num_chunks]
    while chunk_ids and len(chunk_ids) < num_chunks:  # Pad if the table is smaller than requested.
        chunk_ids = chunk_ids + chunk_ids
    chunk_ids = chunk_ids[:num_chunks]

    times = []
    for _ in range(14):
        t0 = time.perf_counter()
        rerank_chunks(RERANK_QUERY, chunk_ids, config=rerank_config)
        times.append((time.perf_counter() - t0) * 1e3)
    times = times[3:]  # Drop warmup, which includes the first ONNX session use.
    return {
        "num_candidates": num_chunks,
        "p50": round(pct(times, 50), 1),
        "p95": round(pct(times, 95), 1),
        "ms_per_candidate": round(pct(times, 50) / num_chunks, 1),
    }


def bench_pipeline(config: Any, query_vectors: list[np.ndarray]) -> dict[str, Any]:
    """End-to-end hybrid retrieval + rerank for two candidate-pool sizes.

    The query is a plain string, so the pipeline would normally call the embedding model. The
    benchmark stubs the query embedder in `install_engine`, so no API credentials are needed.
    """
    from platformdirs import user_data_dir
    from rerankers import Reranker
    from raglite._search import search_and_rerank_chunks, vector_search

    rerank_config = replace(
        config,
        reranker=Reranker(
            "ms-marco-MiniLM-L-12-v2",
            model_type="flashrank",
            verbose=0,
            cache_dir=user_data_dir("raglite", ensure_exists=True),
        ),
    )

    def timeit(fn: Any, runs: int, warmup: int = 3) -> list[float]:
        times = []
        for i in range(runs + warmup):
            t0 = time.perf_counter()
            fn(i)
            times.append((time.perf_counter() - t0) * 1e3)
        return times[warmup:]

    out: dict[str, Any] = {}
    for num_results in (20, 8):
        t = timeit(
            lambda i, n=num_results: search_and_rerank_chunks(
                RERANK_QUERY, num_results=n, config=rerank_config
            ),
            10,
        )
        out[f"hybrid_rerank_num_results_{num_results}"] = {
            "p50": round(pct(t, 50), 1),
            "p95": round(pct(t, 95), 1),
        }
    t = timeit(
        lambda i: vector_search(query_vectors[i % len(query_vectors)], num_results=8, config=config),
        15,
    )
    out["vector_only_num_results_8"] = {"p50": round(pct(t, 50), 1)}
    return out


_BENCH_ENGINE: Any = None


def create_bench_engine() -> Any:
    """Return the pinned benchmark engine (set up by main)."""
    assert _BENCH_ENGINE is not None, "benchmark engine not initialised"
    return _BENCH_ENGINE


def main() -> None:
    """Run the latency diagnosis."""
    global _BENCH_ENGINE  # noqa: PLW0603

    import os

    os.chdir(Path(__file__).parent)
    from raglite import RAGLiteConfig

    if not Path(DB_FILE).exists():
        raise SystemExit(f"{DB_FILE} not found; run fill_and_search.py first.")

    engine = make_engine()
    _BENCH_ENGINE = engine
    install_engine(engine)
    config = RAGLiteConfig(
        db_url=f"duckdb:///{DB_FILE}",
        reranker=None,
        llm="gpt-4o-mini",
        embedder="text-embedding-3-small",
    )

    rng = np.random.default_rng(7)
    query_vectors = [
        (lambda e: e / np.linalg.norm(e))(rng.standard_normal(EMBED_DIM).astype(np.float32))
        for _ in range(NUM_QUERIES)
    ]

    print("=" * 70)
    print("0. INDEX DEFINITION (as stored in the benchmark database)")
    print("=" * 70)
    conn = engine.raw_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT index_name, sql FROM duckdb_indexes() WHERE index_name = 'vector_search_chunk_index';"
    )
    for name, sql in cursor.fetchall():
        print(f"{name}: {sql}")
    cursor.execute("SELECT COUNT(*) FROM chunk_embedding;")
    print(f"chunk_embedding rows: {cursor.fetchone()[0]}")
    cursor.close()

    print("\n" + "=" * 70)
    print("1. QUERY PLAN: does DuckDB use the HNSW index?")
    print("=" * 70)
    plan = bench_plan(engine)
    print(json.dumps(plan, indent=2))

    print("\n" + "=" * 70)
    print("2. VECTOR SEARCH LATENCY")
    print("=" * 70)
    print(json.dumps(bench_vector(engine, config, query_vectors), indent=2))

    print("\n" + "=" * 70)
    print("3. RERANK LATENCY vs CANDIDATE COUNT")
    print("=" * 70)
    reranks = [bench_rerank(config, n) for n in (20, 10, 5)]
    for row in reranks:
        print(row)
    print(f"statistics: {statistics.mean(r['p50'] for r in reranks):.0f} ms mean p50")

    print("\n" + "=" * 70)
    print("4. END-TO-END HYBRID PIPELINE")
    print("=" * 70)
    print(json.dumps(bench_pipeline(config, query_vectors), indent=2))


if __name__ == "__main__":
    main()
