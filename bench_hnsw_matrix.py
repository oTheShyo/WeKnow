"""Which (index metric, distance operator) combinations does DuckDB VSS actually index-serve?

The retrieval path in RAGLite compiles `config.vector_search_distance_metric` into a distance
operator (`_typing.py:_embedding_distance_duckdb`):
    cosine -> array_cosine_distance   dot -> array_negative_inner_product   l2 -> array_distance
and creates the HNSW index with the matching metric (`_database.py:640-645`):
    cosine -> 'cosine'   dot -> 'ip'   l2 -> 'l2sq'

If the operator and the index metric disagree, DuckDB cannot use the index and silently falls
back to a sequential scan. This script builds the same 10k x 1536 table on a *copy* of the
benchmark database, recreates the HNSW index under each metric, and reports for each operator
whether the index is used and what the latency is.

⚠️ CAVEAT (2026-09-28): this script probes the *reduced* query shape
`SELECT chunk_id, <op>(embedding, ?) AS dist FROM chunk_embedding ORDER BY dist LIMIT 200`,
which DOES get `HNSW_INDEX_SCAN`. RAGLite's real `vector_search` additionally projects
`1.0 - <op>(...) AS sim`, and that second distance expression makes DuckDB drop the index scan
and fall back to `SEQ_SCAN`. So "this script says the index is used" does NOT mean the real
retrieval path uses it — see 复现指南-实测数据.md §3.1 for the side-by-side EXPLAIN.

Run:  python bench_hnsw_matrix.py
"""

import json
import shutil
import time
from pathlib import Path

import duckdb
import numpy as np

EMBED_DIM = 1536
NUM_QUERIES = 30
NUM_ROWS = 10_000
SRC_DB = "bench_data.db"
WORK_DB = "bench_hnsw_matrix.db"
OPERATORS = ("array_cosine_distance", "array_distance")


def build_work_db() -> None:
    """Copy the benchmark database so the fixture is never mutated."""
    if not Path(SRC_DB).exists():
        raise SystemExit(f"{SRC_DB} not found; run fill_and_search.py first.")
    shutil.copyfile(SRC_DB, WORK_DB)


def connect() -> duckdb.DuckDBPyConnection:
    """Open the working database with VSS available on a single connection."""
    conn = duckdb.connect(WORK_DB)
    conn.execute("INSTALL vss; LOAD vss;")
    conn.execute("SET hnsw_enable_experimental_persistence = true;")
    conn.execute("SET hnsw_ef_search = 160;")
    return conn


def recreate_index(conn: duckdb.DuckDBPyConnection, metric: str | None) -> None:
    """Drop and recreate the vector index, optionally pinning the distance metric."""
    conn.execute("DROP INDEX IF EXISTS vector_search_chunk_index;")
    with_clause = "" if metric is None else f" WITH (metric = '{metric}')"
    conn.execute(
        f"CREATE INDEX vector_search_chunk_index ON chunk_embedding USING HNSW (embedding){with_clause};"
    )


def uses_hnsw(conn: duckdb.DuckDBPyConnection, operator: str) -> bool:
    """Does the physical plan for this operator mention HNSW?"""
    plan = conn.execute(
        f"EXPLAIN SELECT chunk_id, {operator}(embedding, CAST(? AS FLOAT[{EMBED_DIM}])) AS dist "
        "FROM chunk_embedding ORDER BY dist LIMIT 200;",
        [[0.0] * EMBED_DIM],
    ).fetchall()
    text = "\n".join(str(value) for row in plan for value in row if value)
    return "hnsw" in text.lower()


def bench(conn: duckdb.DuckDBPyConnection, operator: str, vectors: list[np.ndarray]) -> dict:
    """p50/p95 latency for one operator, warmup discarded."""
    sql = (
        f"SELECT chunk_id FROM chunk_embedding ORDER BY {operator}"
        f"(embedding, CAST(? AS FLOAT[{EMBED_DIM}])) LIMIT 200;"
    )
    times = []
    for i in range(NUM_QUERIES + 5):
        t0 = time.perf_counter()
        conn.execute(sql, [vectors[i % len(vectors)].tolist()]).fetchall()
        times.append((time.perf_counter() - t0) * 1e3)
    times = times[5:]
    return {
        "p50_ms": round(float(np.percentile(times, 50)), 1),
        "p95_ms": round(float(np.percentile(times, 95)), 1),
    }


def main() -> None:
    """Build the (metric, operator) matrix."""
    import os

    os.chdir(Path(__file__).parent)
    build_work_db()
    conn = connect()
    num_rows = conn.execute("SELECT COUNT(*) FROM chunk_embedding;").fetchone()
    col_type = conn.execute(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name = 'chunk_embedding' AND column_name = 'embedding';"
    ).fetchone()
    print(f"rows: {num_rows[0]}")
    print(f"embedding column type: {col_type[0]}")

    rng = np.random.default_rng(11)
    vectors = [
        (lambda e: e / np.linalg.norm(e))(rng.standard_normal(EMBED_DIM).astype(np.float32))
        for _ in range(NUM_QUERIES)
    ]

    matrix: dict[str, dict] = {}
    # `None` reproduces the benchmark fixture, which created the index without a WITH clause.
    for metric in (None, "l2sq", "cosine", "ip"):
        label = "default(no WITH clause)" if metric is None else metric
        recreate_index(conn, metric)
        matrix[label] = {}
        for operator in OPERATORS:
            matrix[label][operator] = {
                "hnsw_used": uses_hnsw(conn, operator),
                **bench(conn, operator, vectors),
            }
    conn.close()
    Path(WORK_DB).unlink(missing_ok=True)
    print(json.dumps(matrix, indent=2))


if __name__ == "__main__":
    main()
