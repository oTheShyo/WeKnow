"""File-backed vs in-memory vector search, on the same 10k x 1536 table.

The documented 46.5 ms figure was measured against an *in-memory* DuckDB table. A file-backed
database additionally reads the vector column off disk, so its latency depends on whether the
page cache is warm. This script isolates that difference with nothing else running.

It deliberately uses the plain `duckdb` module only -- no SQLAlchemy -- because DuckDB refuses to
attach the same database file twice within one process, and the other diagnostics hold a
SQLAlchemy connection to it.

Run:  python bench_mem_vs_file.py
"""

import json
import time
from pathlib import Path

import duckdb
import numpy as np

EMBED_DIM = 1536
NUM_QUERIES = 30
DB_FILE = "bench_data.db"
QUERY = (
    "SELECT chunk_id, array_cosine_distance(embedding, "
    f"CAST(? AS FLOAT[{EMBED_DIM}])) AS dist "
    "FROM chunk_embedding ORDER BY dist LIMIT 200;"
)


def p50(conn: duckdb.DuckDBPyConnection, vectors: list[np.ndarray], runs: int, warmup: int) -> float:
    """Median wall-clock latency in ms, warmup discarded."""
    times = []
    for i in range(runs + warmup):
        t0 = time.perf_counter()
        conn.execute(QUERY, [vectors[i % len(vectors)].tolist()]).fetchall()
        times.append((time.perf_counter() - t0) * 1e3)
    return round(float(np.percentile(times[warmup:], 50)), 1)


def main() -> None:
    """Measure the same query against a file database, an in-memory copy, and memory w/o index."""
    import os

    os.chdir(Path(__file__).parent)
    rng = np.random.default_rng(3)
    vectors = [
        (lambda e: e / np.linalg.norm(e))(rng.standard_normal(EMBED_DIM).astype(np.float32))
        for _ in range(NUM_QUERIES)
    ]
    out: dict[str, float | str] = {}

    # 1. File-backed, database carried over from fill_and_search.py (index already built).
    file_conn = duckdb.connect(DB_FILE)
    file_conn.execute("INSTALL vss; LOAD vss;")
    file_conn.execute("SET hnsw_enable_experimental_persistence = true;")
    file_conn.execute("SET hnsw_ef_search = 160;")
    out["file_backed_hnsw"] = p50(file_conn, vectors, runs=NUM_QUERIES, warmup=5)
    file_conn.close()

    # 2. In-memory copy of the same table, HNSW index rebuilt in memory.
    mem = duckdb.connect(":memory:")
    mem.execute("INSTALL vss; LOAD vss;")
    mem.execute(f"ATTACH '{DB_FILE}' AS src (READ_ONLY);")
    mem.execute("CREATE TABLE chunk_embedding AS SELECT * FROM src.chunk_embedding;")
    mem.execute("DETACH src;")
    mem.execute("SET hnsw_enable_experimental_persistence = true;")
    mem.execute("SET hnsw_ef_search = 160;")
    mem.execute(
        "CREATE INDEX vector_search_chunk_index ON chunk_embedding USING HNSW (embedding) "
        "WITH (metric = 'cosine');"
    )
    out["in_memory_hnsw"] = p50(mem, vectors, runs=NUM_QUERIES, warmup=5)

    # 3. Same in-memory table with no index at all: what the ANN index actually buys at 10k.
    mem.execute("DROP INDEX vector_search_chunk_index;")
    out["in_memory_no_index"] = p50(mem, vectors, runs=NUM_QUERIES, warmup=5)
    mem.close()

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
