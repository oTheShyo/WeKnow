"""Rebuild the chunk projection, then attribute `graph_search`'s ~6.5s with cProfile.

The density sweep showed the cost is independent of the projection size, so the neighbour SQL is
not it. A profiler settles the question instead of another guess.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from pathlib import Path

import ir_datasets
from platformdirs import user_data_dir

from weknow._config import RAGLiteConfig
from weknow._database import create_database_engine
from weknow._graph import CooccurrenceExtractor, GraphConfig, build_chunk_relations, graph_search

DB_NAME = "graph_beir-nfcorpus.db"
CAP = 32

db_path = Path(user_data_dir("weknow", ensure_exists=True)) / DB_NAME
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
        max_chunks_per_entity=CAP,
    ),
)

print(f"rebuilding projection at cap={CAP} ...", flush=True)
start = time.perf_counter()
edges = build_chunk_relations(config=config)
print(f"  {edges:,} edges in {time.perf_counter() - start:.2f}s", flush=True)

dataset = ir_datasets.load("beir/nfcorpus/test")
queries = [query.text for query in dataset.queries_iter()][:3]

profiler = cProfile.Profile()
start = time.perf_counter()
profiler.enable()
for query in queries:
    graph_search(query, num_results=200, config=config)
profiler.disable()
print(f"3 queries in {time.perf_counter() - start:.3f}s\n", flush=True)

stream = io.StringIO()
pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(28)
for line in stream.getvalue().splitlines():
    if line.strip():
        print(line, flush=True)
