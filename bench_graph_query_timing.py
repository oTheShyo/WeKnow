"""Time `graph_search` (and its parts) on the already-built NFCorpus graph.

Purpose: before committing to a 6-arm x 323-query ablation, measure the real per-query cost so the
total runtime is known rather than guessed. Runs against `graph_beir-nfcorpus.db` read-only, so it
is safe to run while the ablation is not.

    .venv/Scripts/python.exe bench_graph_query_timing.py [num_queries] [graph_chunk_relation.db]
"""

import statistics
import sys
import time
from pathlib import Path

import duckdb
import ir_datasets
from bench_beir import BASE_KWARGS

from weknow._config import RAGLiteConfig
from weknow._graph import CooccurrenceExtractor, GraphConfig, graph_search
from weknow._search import keyword_search

NUM_QUERIES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
DB_NAME = sys.argv[2] if len(sys.argv) > 2 else "graph_beir-nfcorpus.db"
DB_PATH = Path("C:/Users/HP/AppData/Local/raglite/raglite") / DB_NAME
POOL = 200  # 2 x 100, matching the ablation's oversample

read_only = duckdb.connect(DB_PATH.as_posix(), read_only=True)
edges = read_only.execute("SELECT count(*) FROM graph_chunk_relation").fetchone()[0]
read_only.close()
print(f"db       : {DB_PATH.name}")
print(f"edges    : {edges:,}")

config = RAGLiteConfig(
    **BASE_KWARGS,
    db_url=f"duckdb:///{DB_PATH.as_posix()}",
    graph=GraphConfig(enabled=True, extractor=CooccurrenceExtractor(), build_chunk_graph=True),
)

dataset = ir_datasets.load("beir/nfcorpus/test")
queries = [q.text for q in dataset.queries_iter()][:NUM_QUERIES]

keyword_times: list[float] = []
graph_times: list[float] = []
hits = 0
for index, query in enumerate(queries, start=1):
    start = time.perf_counter()
    keyword_search(query, num_results=POOL, config=config)
    keyword_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    graph_ids, _ = graph_search(query, num_results=POOL, config=config)
    graph_times.append(time.perf_counter() - start)
    hits += 1 if graph_ids else 0
    print(f"  q{index:3d}  keyword={keyword_times[-1] * 1000:8.1f}ms  graph={graph_times[-1]:7.3f}s")


def report(name: str, values: list[float]) -> None:
    ordered = sorted(values)
    p50 = statistics.median(ordered)
    p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
    print(f"{name:8s}  mean={statistics.fmean(values):8.3f}s  p50={p50:8.3f}s  p90={p90:8.3f}s")


print()
report("keyword", keyword_times)
report("graph", graph_times)
print(f"graph hit {hits}/{len(queries)}")
print()
print(f"projected 6-arm x 323-query runtime: {5 * 323 * statistics.fmean(graph_times) / 60:.1f} min "
      f"(5 graph arms, ignoring the keyword arm)")
