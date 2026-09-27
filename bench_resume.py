"""Resume benchmark: measure RAGLite chunking, insertion, and search latency.

Embedding models are mocked (random unit vectors) so that the measurements cover
the pure software pipeline: PDF -> Markdown -> sentences -> chunklets -> BIP
semantic chunking -> parallel insertion -> DuckDB HNSW/FTS search -> RRF ->
rerank. Model inference latency is explicitly NOT measured.
"""

import json
import os
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

EMBED_DIM = 1536  # text-embedding-3-small dimension.
NUM_DOCS = 24  # Copies of the NVIDIA report to build a 10k+-vector database.
NUM_QUERIES = 50
NVIDIA_PDF = Path(__file__).parent / "tests" / "NVIDIA-report.pdf"
SPECREL_PDF = Path(__file__).parent / "tests" / "specrel.pdf"
OUT_JSON = Path(__file__).parent / "bench_results.json"

# Explicit LLM/embedder strings to avoid the llama.cpp lazy-import default factories.
BASE_KWARGS = {"llm": "gpt-4o-mini", "embedder": "text-embedding-3-small"}


def mock_embeddings(strings: list[str], **_kwargs) -> np.ndarray:
    """Random unit-norm embeddings (one row per input string)."""
    rng = np.random.default_rng()
    embeddings = rng.standard_normal((len(strings), EMBED_DIM)).astype(np.float32)
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


# Precomputed real sentence split, so the insertion benchmark measures the chunklet/BIP/DB
# pipeline without the SaT ONNX inference (which is timed separately in bench_chunking).
_PRECOMPUTED_SENTENCES: list[str] | None = None


def _cached_split_sentences(content: str, max_len: int = 2048) -> list[str]:
    assert _PRECOMPUTED_SENTENCES is not None
    return list(_PRECOMPUTED_SENTENCES)


def pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def bench_chunking(markdown: str) -> dict:
    """Stage-by-stage timing of the real chunking pipeline (no embedding model)."""
    from raglite._split_chunklets import split_chunklets
    from raglite._split_chunks import split_chunks
    from raglite._split_sentences import split_sentences

    results = {}
    for run in range(5):
        timings = {}
        t0 = time.perf_counter()
        sentences = split_sentences(markdown, max_len=2048)
        timings["split_sentences_ms"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        chunklets = split_chunklets(sentences, max_size=2048)
        timings["split_chunklets_ms"] = (time.perf_counter() - t0) * 1e3
        # Deterministic chunklet embeddings for the BIP cost function.
        rng = np.random.default_rng(42)
        chunklet_embeddings = rng.standard_normal((len(chunklets), EMBED_DIM)).astype(np.float32)
        chunklet_embeddings /= np.linalg.norm(chunklet_embeddings, axis=1, keepdims=True)
        t0 = time.perf_counter()
        chunks, _ = split_chunks(chunklets, chunklet_embeddings, max_size=2048)
        timings["split_chunks_bip_ms"] = (time.perf_counter() - t0) * 1e3
        results[run] = {
            **timings,
            "num_sentences": len(sentences),
            "num_chunklets": len(chunklets),
            "num_chunks": len(chunks),
        }
    med = {
        key: statistics.median(results[run][key] for run in results)
        for key in results[0]
        if key != "num_sentences" and key != "num_chunklets" and key != "num_chunks"
    }
    return {
        "doc_chars": len(markdown),
        "num_sentences": results[0]["num_sentences"],
        "num_chunklets": results[0]["num_chunklets"],
        "num_chunks": results[0]["num_chunks"],
        "median_ms": {k: round(v, 1) for k, v in med.items()},
        "total_chunking_median_ms": round(sum(med.values()), 1),
    }


def bench_insert(markdown: str) -> dict:
    """insert_documents with mocked embeddings: workers=1 vs workers=4."""
    import raglite._insert as insert_module
    from raglite import Document, RAGLiteConfig
    from raglite._split_sentences import split_sentences as real_split_sentences

    global _PRECOMPUTED_SENTENCES
    _PRECOMPUTED_SENTENCES = real_split_sentences(markdown, max_len=2048)

    documents = [
        Document.from_text(markdown + f"\n\nAppendix variant {i}.", id=f"nvda-variant-{i}")
        for i in range(NUM_DOCS)
    ]
    results = {}
    for max_workers in (1, 4):
        db_url = f"duckdb:///bench_insert_w{max_workers}.db"
        config = RAGLiteConfig(db_url=db_url, reranker=None, **BASE_KWARGS)
        # Patch the embedding functions and the sentence splitter used inside insert_documents.
        original = (
            insert_module.embed_strings,
            insert_module.embed_strings_without_late_chunking,
            insert_module.split_sentences,
        )
        insert_module.embed_strings = mock_embeddings
        insert_module.embed_strings_without_late_chunking = mock_embeddings
        insert_module.split_sentences = _cached_split_sentences
        try:
            t0 = time.perf_counter()
            insert_module.insert_documents(documents, max_workers=max_workers, config=config)
            elapsed = time.perf_counter() - t0
        finally:
            (
                insert_module.embed_strings,
                insert_module.embed_strings_without_late_chunking,
                insert_module.split_sentences,
            ) = original
        db_size_mb = Path(db_url.removeprefix("duckdb:///")).stat().st_size / 1e6
        # Count chunks and chunk embeddings.
        from raglite._database import Chunk, ChunkEmbedding, create_database_engine
        from sqlmodel import Session, func, select

        with Session(create_database_engine(config)) as session:
            num_chunks = session.exec(select(func.count()).select_from(Chunk)).one()
            num_embeddings = session.exec(select(func.count()).select_from(ChunkEmbedding)).one()
        results[max_workers] = {
            "wall_s": round(elapsed, 2),
            "num_chunks": int(num_chunks),
            "num_chunk_embeddings": int(num_embeddings),
            "db_size_mb": round(db_size_mb, 1),
            "max_workers": max_workers,
        }
        print(f"insert max_workers={max_workers}: {results[max_workers]}")
    return results


def bench_search() -> dict:
    """Search latency on the 4-worker database (~10k multi-vectors)."""
    from raglite import RAGLiteConfig
    from raglite._database import create_database_engine
    from raglite._search import (
        keyword_search,
        reciprocal_rank_fusion,
        retrieve_chunk_spans,
        rerank_chunks,
        vector_search,
    )
    from sqlmodel import Session, func, select
    from raglite._database import Chunk, ChunkEmbedding

    db_url = "duckdb:///bench_insert_w4.db"
    config = RAGLiteConfig(db_url=db_url, reranker=None, **BASE_KWARGS)
    with Session(create_database_engine(config)) as session:
        num_chunks = int(session.exec(select(func.count()).select_from(Chunk)).one())
        num_embeddings = int(session.exec(select(func.count()).select_from(ChunkEmbedding)).one())

    rng = np.random.default_rng(7)
    query_vectors = [
        (lambda e: e / np.linalg.norm(e))(
            rng.standard_normal(EMBED_DIM).astype(np.float32)
        )
        for _ in range(NUM_QUERIES)
    ]

    # --- Vector search (HNSW via DuckDB VSS), query passed as a vector (no model). ---
    vector_times = []
    for i in range(NUM_QUERIES + 5):
        t0 = time.perf_counter()
        chunk_ids, _ = vector_search(query_vectors[i % NUM_QUERIES], num_results=20, config=config)
        vector_times.append((time.perf_counter() - t0) * 1e3)
    vector_times = vector_times[5:]  # Drop warmup.

    # --- Keyword search (BM25 FTS), real text query. ---
    keyword_times = []
    for i in range(NUM_QUERIES):
        query = [
            "data center revenue growth drivers",
            "gaming segment operating income",
            "gross margin outlook next quarter",
            "supply inventory purchase commitments",
            "segment revenue geographic breakdown",
        ][i % 5]
        t0 = time.perf_counter()
        ks_ids, _ = keyword_search(query, num_results=20, config=config)
        keyword_times.append((time.perf_counter() - t0) * 1e3)

    # --- RRF fusion over precomputed rankings (mirrors hybrid_search). ---
    rrf_times = []
    vs_ids, _ = vector_search(query_vectors[0], num_results=40, config=config)
    ks_ids, _ = keyword_search("data center revenue growth drivers", num_results=40, config=config)
    for _ in range(200):
        t0 = time.perf_counter()
        reciprocal_rank_fusion([vs_ids, ks_ids], weights=[0.75, 0.25])
        rrf_times.append((time.perf_counter() - t0) * 1e3)

    # --- Chunk spans (neighbor extension + grouping). ---
    span_times = []
    for i in range(20):
        t0 = time.perf_counter()
        spans = retrieve_chunk_spans(vs_ids[:8], config=config)
        span_times.append((time.perf_counter() - t0) * 1e3)

    # --- Cross-encoder rerank (FlashRank, downloads a small local model). ---
    rerank_times, reranker_ok = [], True
    try:
        from rerankers import Reranker

        rerank_config = replace(
            config, reranker=Reranker("ms-marco-MiniLM-L-12-v2", model_type="flashrank", verbose=0)
        )
        for i in range(20):
            t0 = time.perf_counter()
            reranked = rerank_chunks(
                "data center revenue growth drivers", vs_ids[:20], config=rerank_config
            )
            rerank_times.append((time.perf_counter() - t0) * 1e3)
    except Exception as error:  # noqa: BLE001
        reranker_ok = False
        print(f"reranker skipped: {error}")

    results = {
        "num_chunks": num_chunks,
        "num_chunk_embeddings": num_embeddings,
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
        "num_span_chunks_last_run": len(spans),
        "num_reranked_last_run": len(reranked) if reranker_ok else 0,
    }
    return results


def main() -> None:
    os.chdir(Path(__file__).parent)
    from raglite import RAGLiteConfig
    from raglite._markdown import document_to_markdown

    print("Converting PDF to Markdown...")
    t0 = time.perf_counter()
    paper_markdown = document_to_markdown(
        SPECREL_PDF, config=RAGLiteConfig(reranker=None, **BASE_KWARGS)
    )
    pdf_to_md_s = time.perf_counter() - t0
    # Duplicate the paper into a ~100k-character document (ten-characters-of-ten-thousands scale).
    markdown = paper_markdown + "\n\n" + paper_markdown
    print(f"PDF -> Markdown: {len(paper_markdown)} chars; benchmark document: {len(markdown)} chars")

    print("Benchmarking chunking pipeline...")
    chunking = bench_chunking(markdown)
    print(chunking)

    print("Benchmarking insertion (mocked embeddings)...")
    insertion = bench_insert(markdown)

    print("Benchmarking search latency...")
    search = bench_search()

    output = {
        "environment": {
            "os": platform.platform(),
            "cpu": f"{os.cpu_count()} logical cores, {platform.processor()}",
            "python": platform.python_version(),
            "note": "Embedding model mocked; measurements cover the software pipeline only.",
        },
        "pdf_to_markdown_s": round(pdf_to_md_s, 1),
        "chunking": chunking,
        "insertion": insertion,
        "search": search,
    }
    OUT_JSON.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
