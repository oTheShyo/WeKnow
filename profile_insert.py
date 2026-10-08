"""Profile a single-document insertion to locate the bottleneck."""

import cProfile
import pstats
import sys
from pathlib import Path

import weknow._insert as insert_module
from weknow import Document, RAGLiteConfig

sys.path.insert(0, str(Path(__file__).parent))
from bench_resume import BASE_KWARGS, _cached_split_sentences, mock_embeddings  # noqa: E402
from weknow._split_sentences import split_sentences as real_split_sentences  # noqa: E402

import bench_resume  # noqa: E402

markdown_path = Path(__file__).parent / "bench_markdown.txt"
markdown = markdown_path.read_text(encoding="utf-8")
bench_resume._PRECOMPUTED_SENTENCES = real_split_sentences(markdown, max_len=2048)

config = RAGLiteConfig(db_url="duckdb:///bench_probe.db", reranker=None, **BASE_KWARGS)
insert_module.embed_strings = mock_embeddings
insert_module.embed_strings_without_late_chunking = mock_embeddings
insert_module.split_sentences = _cached_split_sentences

doc = Document.from_text(markdown, id="probe-doc")

profiler = cProfile.Profile()
profiler.enable()
insert_module.insert_documents([doc], max_workers=1, config=config)
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats("cumulative").print_stats(18)
