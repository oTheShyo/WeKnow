"""Which SQL shape makes the bounded neighbour lookup cheap?

`_chunk_neighbors` currently ranks every edge with a window function and filters below it. If DuckDB
computes that window over the whole table before the `IN` filter applies, each batch pays a full scan
of the 1.95M-edge table. A LATERAL join per chunk should instead be an index lookup plus a top-K.

Compared at a realistic frontier of 1,400 chunks:
  A. window subquery + IN
  B. QUALIFY + IN
  C. LATERAL join with per-chunk ORDER BY ... LIMIT
  D. plain IN, no bound (what the code did before the bound existed)
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
from platformdirs import user_data_dir

db_path = Path(user_data_dir("raglite", ensure_exists=True)) / "graph_beir-nfcorpus.db"
connection = duckdb.connect(db_path.as_posix(), read_only=True)
edges = connection.execute("SELECT count(*) FROM graph_chunk_relation").fetchone()[0]
print(f"edges: {edges:,}", flush=True)
ids = [r[0] for r in connection.execute(
    "SELECT DISTINCT source_chunk_id FROM graph_chunk_relation LIMIT 1400").fetchall()]
print(f"frontier: {len(ids)}\n", flush=True)

A = """SELECT source_chunk_id, target_chunk_id, weight FROM (
        SELECT source_chunk_id, target_chunk_id, weight,
               row_number() OVER (PARTITION BY source_chunk_id ORDER BY weight DESC) AS rk
        FROM graph_chunk_relation WHERE source_chunk_id IN ({ph})
    ) t WHERE rk <= 32"""
B = """SELECT source_chunk_id, target_chunk_id, weight FROM graph_chunk_relation
    WHERE source_chunk_id IN ({ph})
    QUALIFY row_number() OVER (PARTITION BY source_chunk_id ORDER BY weight DESC) <= 32"""
C = """SELECT f.source_chunk_id, n.target_chunk_id, n.weight
    FROM (SELECT unnest(?) AS source_chunk_id) AS f
    JOIN LATERAL (SELECT target_chunk_id, weight FROM graph_chunk_relation
                  WHERE source_chunk_id = f.source_chunk_id
                  ORDER BY weight DESC LIMIT 32) AS n ON true"""
D = """SELECT source_chunk_id, target_chunk_id, weight FROM graph_chunk_relation
    WHERE source_chunk_id IN ({ph})"""

def run(label, sql, params):
    start = time.perf_counter()
    rows = connection.execute(sql, params).fetchall()
    el = time.perf_counter() - start
    print(f"  {label:34s} {el:7.3f}s  {len(rows):8,d} rows", flush=True)

for batch_size in (256, 1400):
    print(f"batch={batch_size}")
    for label, sql in (("A window subquery", A), ("B QUALIFY", B), ("D plain IN", D)):
        total = 0.0; rows = 0
        for off in range(0, len(ids), batch_size):
            batch = ids[off:off + batch_size]
            ph = ",".join("?" * len(batch))
            start = time.perf_counter()
            got = connection.execute(sql.format(ph=ph), batch).fetchall()
            total += time.perf_counter() - start
            rows += len(got)
        print(f"  {label:34s} {total:7.3f}s  {rows:8,d} rows", flush=True)
    run("C lateral (one statement)", C, [ids])
connection.close()
