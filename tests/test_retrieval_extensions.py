"""Tests for the retrieval extensions.

Covers the retrieval result cache, incremental document sync, intent routing, low-confidence
secondary retrieval, and generation-side evaluation metrics.
"""

import time
from typing import Any

import pytest

from raglite import Document
from raglite._cache import RetrievalCache, get_retrieval_cache
from raglite._eval_generation import _tokenize, bleu_n, evaluate_generation, rouge_l_f1
from raglite._iterative_search import search_with_secondary_retrieval
from raglite._router import route_search_method

NUM_RESULTS = 4  # Results requested in the secondary retrieval tests.
OVERSAMPLE = 4  # Oversampling factor in the secondary retrieval tests.
DISJOINT_BLEU_CEILING = 0.5  # BLEU ceiling for disjoint texts under add-one smoothing.
PARTIAL_ROUGE_FLOOR = 0.5  # ROUGE-L floor for partially overlapping texts.

# ---------------------------------------------------------------------------
# Generation evaluation metrics.
# ---------------------------------------------------------------------------


def test_tokenize() -> None:
    """CJK characters are tokenized individually and words are lowercased."""
    assert _tokenize("Hello World") == ["hello", "world"]
    assert _tokenize("量子纠错") == ["量", "子", "纠", "错"]
    assert _tokenize("Shor 算法") == ["shor", "算", "法"]


def test_rouge_l_f1() -> None:
    """Identical texts score 1.0, disjoint texts score 0.0, partial overlap is in between."""
    assert rouge_l_f1("量子纠错是关键难题", "量子纠错是关键难题") == 1.0
    assert rouge_l_f1("量子纠错", "经典比特") == 0.0
    reference = "quantum error correction protects fragile quantum states"
    hypothesis = "quantum error correction protects the fragile states"
    score = rouge_l_f1(reference, hypothesis)
    assert PARTIAL_ROUGE_FLOOR < score < 1.0
    assert rouge_l_f1("", "anything") == 0.0
    assert rouge_l_f1("anything", "") == 0.0


def test_bleu_n() -> None:
    """Identical texts score close to 1.0, disjoint texts stay low under add-one smoothing."""
    text = "Shor 算法能够在多项式时间内分解大整数"
    assert bleu_n(text, text) == pytest.approx(1.0)
    # Disjoint texts: zero matches at every n-gram order; with add-one smoothing the geometric
    # mean stays low (here ~0.30 due to the 4-gram order contributing 1/2), far below 1.0.
    assert bleu_n("量子纠错", "经典比特") < DISJOINT_BLEU_CEILING
    assert bleu_n("", "anything") == 0.0


def test_evaluate_generation_with_pregenerated_answers() -> None:
    """evaluate_generation scores pre-generated answers without touching the database."""
    qa_pairs = [("什么是量子比特?", "量子比特是量子计算的基本信息单元")]
    result = evaluate_generation(qa_pairs, answers=["量子比特是量子计算的基本信息单元"])
    assert result["n"] == 1
    assert result["mean_rouge_l"] == 1.0
    assert result["mean_bleu"] == pytest.approx(1.0)


def test_evaluate_generation_requires_answers_or_config() -> None:
    """Without pre-generated answers or a config, evaluation raises."""
    with pytest.raises(ValueError, match="answers"):
        evaluate_generation([("q", "reference")])


# ---------------------------------------------------------------------------
# Retrieval result cache.
# ---------------------------------------------------------------------------


def test_retrieval_cache_set_get_clear() -> None:
    """Cache entries are returned, invalidated on clear, and keyed stably."""
    cache = RetrievalCache()
    key = cache.make_key("量子比特", num_results=8, method_name="hybrid_search")
    cache.set(key, (["c1", "c2"], [0.9, 0.8]))
    assert cache.get(key) == (["c1", "c2"], [0.9, 0.8])
    cache.clear()
    assert cache.get(key) is None


def test_retrieval_cache_ttl_expiry() -> None:
    """Entries expire after the TTL."""
    cache = RetrievalCache(ttl=0.01)
    key = cache.make_key("query", num_results=3)
    cache.set(key, (["c1"], [1.0]))
    assert cache.get(key) == (["c1"], [1.0])
    time.sleep(0.02)
    assert cache.get(key) is None


def test_retrieval_cache_max_size_eviction() -> None:
    """When full, the entry with the earliest expiry is evicted."""
    cache = RetrievalCache(max_size=2)
    key_a = cache.make_key("a", num_results=1)
    key_b = cache.make_key("b", num_results=1)
    key_c = cache.make_key("c", num_results=1)
    cache.set(key_a, (["ca"], [1.0]))
    time.sleep(0.005)
    cache.set(key_b, (["cb"], [1.0]))
    cache.set(key_c, (["cc"], [1.0]))  # Evicts key_a (earliest expiry).
    assert cache.get(key_a) is None
    assert cache.get(key_b) is not None
    assert cache.get(key_c) is not None


def test_retrieval_cache_make_key_metadata_normalization() -> None:
    """The same metadata filter in different orders maps to the same key."""
    key_1 = RetrievalCache.make_key("q", num_results=3, metadata_filter={"a": [1, 2], "b": "x"})
    key_2 = RetrievalCache.make_key("q", num_results=3, metadata_filter={"b": "x", "a": [2, 1]})
    assert key_1 == key_2
    key_3 = RetrievalCache.make_key("q", num_results=3, metadata_filter={"a": [1, 3]})
    assert key_1 != key_3


# ---------------------------------------------------------------------------
# Intent routing.
# ---------------------------------------------------------------------------


def test_route_search_method_heuristics() -> None:
    """Keyword-heavy short queries and quoted phrases go to BM25; the rest to hybrid."""
    keyword_method = route_search_method('"Shor 算法" 复杂度')
    assert keyword_method.__name__ == "keyword_search"
    keyword_method = route_search_method("Grover 平方根 加速")
    assert keyword_method.__name__ == "keyword_search"
    hybrid_method = route_search_method("请用一句话解释什么是量子比特以及它和经典比特的区别")
    assert hybrid_method.__name__ == "hybrid_search"


# ---------------------------------------------------------------------------
# Low-confidence secondary retrieval.
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_search() -> dict[str, Any]:
    """Stub search method returning a small ordered candidate pool, recording calls."""
    calls: list[int] = []

    def fake_search(
        query: str,
        *,
        num_results: int,
        metadata_filter: Any = None,
        config: Any = None,
    ) -> tuple[list[str], list[float]]:
        calls.append(num_results)
        chunk_ids = [f"chunk-{i}" for i in range(num_results)]
        scores = [1.0 - i * 0.01 for i in range(num_results)]
        return chunk_ids, scores

    return {"search": fake_search, "calls": calls}


@pytest.fixture
def test_config() -> Any:
    """A config without model dependencies (no reranker, explicit llm/embedder strings)."""
    from raglite import RAGLiteConfig

    return RAGLiteConfig(
        db_url="duckdb:///:memory:",
        llm="gpt-4o-mini",
        embedder="text-embedding-3-small",
        reranker=None,
    )


def _chunk_like(chunk_ids: list[str]) -> list[Any]:
    """Chunk-like objects with an id attribute, matching the rerank contract."""
    from types import SimpleNamespace

    return [SimpleNamespace(id=chunk_id) for chunk_id in chunk_ids]


def test_secondary_retrieval_skipped_without_reranker(
    mock_search: dict[str, Any], monkeypatch: pytest.MonkeyPatch, test_config: Any
) -> None:
    """Without a reranker there is no confidence signal, so only one search pass runs."""

    def no_reranker_scores(
        query: str, chunk_ids: list[str], *, config: Any = None
    ) -> tuple[list[str], list[float] | None]:
        return _chunk_like(chunk_ids), None

    import raglite._search as search_module

    monkeypatch.setattr(search_module, "rerank_chunks_with_scores", no_reranker_scores)
    chunk_ids, _ = search_with_secondary_retrieval(
        "query", num_results=NUM_RESULTS, search=mock_search["search"], config=test_config
    )
    assert len(chunk_ids) == NUM_RESULTS
    assert mock_search["calls"] == [OVERSAMPLE * NUM_RESULTS]  # Only the first pass ran.


def test_secondary_retrieval_triggered_below_threshold(
    mock_search: dict[str, Any], monkeypatch: pytest.MonkeyPatch, test_config: Any
) -> None:
    """A low top rerank score triggers a second pass with an expanded pool."""

    def low_confidence_scores(
        query: str, chunk_ids: list[str], *, config: Any = None
    ) -> tuple[list[str], list[float] | None]:
        scores = [0.1 - i * 0.001 for i in range(len(chunk_ids))]
        return _chunk_like(chunk_ids), scores

    import raglite._search as search_module

    monkeypatch.setattr(search_module, "rerank_chunks_with_scores", low_confidence_scores)
    chunk_ids, scores = search_with_secondary_retrieval(
        "query",
        num_results=NUM_RESULTS,
        search=mock_search["search"],
        config=test_config,
    )
    # Two passes: the initial pool and the expanded pool (multiplied by 2).
    assert mock_search["calls"] == [OVERSAMPLE * NUM_RESULTS, 2 * OVERSAMPLE * NUM_RESULTS]
    # The best-scoring candidates of both passes are returned.
    assert len(chunk_ids) == NUM_RESULTS
    assert scores == sorted(scores, reverse=True)


def test_secondary_retrieval_not_triggered_above_threshold(
    mock_search: dict[str, Any], monkeypatch: pytest.MonkeyPatch, test_config: Any
) -> None:
    """A high top rerank score skips the second pass."""

    def high_confidence_scores(
        query: str, chunk_ids: list[str], *, config: Any = None
    ) -> tuple[list[str], list[float] | None]:
        scores = [0.9 - i * 0.01 for i in range(len(chunk_ids))]
        return _chunk_like(chunk_ids), scores

    import raglite._search as search_module

    monkeypatch.setattr(search_module, "rerank_chunks_with_scores", high_confidence_scores)
    search_with_secondary_retrieval(
        "query", num_results=NUM_RESULTS, search=mock_search["search"], config=test_config
    )
    assert mock_search["calls"] == [OVERSAMPLE * NUM_RESULTS]  # Only the first pass ran.


# ---------------------------------------------------------------------------
# Incremental document sync (with stubbed insert/delete to avoid model downloads).
# ---------------------------------------------------------------------------


def test_sync_documents_new_and_unchanged(
    monkeypatch: pytest.MonkeyPatch, test_config: Any
) -> None:
    """New documents are inserted; documents already in the database are unchanged."""
    import raglite._incremental as incremental_module

    inserted_ids: list[str] = []
    deleted_ids: list[str] = []

    def fake_insert(documents: list[Document], **kwargs: Any) -> None:
        inserted_ids.extend(doc.id for doc in documents)

    def fake_delete(document_ids: list[str], **kwargs: Any) -> int:
        deleted_ids.extend(document_ids)
        return len(document_ids)

    # Database contains only doc-a (id is the content hash).
    stored_ids = {"doc-a"}

    def fake_existing(session: Any, key: str, key_values: list[str]) -> dict[str, list[Document]]:
        return {
            value: [Document(id=doc_id, filename=f"{doc_id}.md", content="")]
            for value in key_values
            if (doc_id := value) in stored_ids
        }

    monkeypatch.setattr("raglite._insert.insert_documents", fake_insert)
    monkeypatch.setattr("raglite._delete.delete_documents", fake_delete)
    monkeypatch.setattr(incremental_module, "_existing_documents_by_key", fake_existing)
    documents = [
        Document(id="doc-a", filename="doc-a.md", content="unchanged content"),
        Document(id="doc-b", filename="doc-b.md", content="new content"),
    ]
    report = incremental_module.sync_documents(documents, config=test_config)
    assert report["unchanged"] == ["doc-a"]
    assert report["inserted"] == ["doc-b"]
    assert report["deleted"] == []
    assert inserted_ids == ["doc-b"]
    assert deleted_ids == []


def test_sync_documents_hot_update_replaces_stale_version(
    monkeypatch: pytest.MonkeyPatch, test_config: Any
) -> None:
    """A changed document (same filename, new content hash) replaces its stale version."""
    import raglite._incremental as incremental_module

    inserted_ids: list[str] = []
    deleted_ids: list[str] = []

    def fake_insert(documents: list[Document], **kwargs: Any) -> None:
        inserted_ids.extend(doc.id for doc in documents)

    def fake_delete(document_ids: list[str], **kwargs: Any) -> int:
        deleted_ids.extend(document_ids)
        return len(document_ids)

    # Database has doc-v1 under filename "report.md"; the new version has hash doc-v2.
    def fake_existing(session: Any, key: str, key_values: list[str]) -> dict[str, list[Document]]:
        if "report.md" in key_values:
            return {"report.md": [Document(id="doc-v1", filename="report.md", content="")]}

        return {}

    monkeypatch.setattr("raglite._insert.insert_documents", fake_insert)
    monkeypatch.setattr("raglite._delete.delete_documents", fake_delete)
    monkeypatch.setattr(incremental_module, "_existing_documents_by_key", fake_existing)
    documents = [Document(id="doc-v2", filename="report.md", content="updated content")]
    report = incremental_module.sync_documents(documents, key="filename", config=test_config)
    assert report["unchanged"] == []
    assert report["inserted"] == ["doc-v2"]
    assert report["deleted"] == ["doc-v1"]
    assert inserted_ids == ["doc-v2"]
    assert deleted_ids == ["doc-v1"]


def test_sync_documents_invalidates_cache(
    monkeypatch: pytest.MonkeyPatch, test_config: Any
) -> None:
    """Syncing documents clears the process-wide retrieval cache."""
    import raglite._incremental as incremental_module

    def fake_insert(documents: list[Document], **kwargs: Any) -> None:
        pass

    def fake_delete(document_ids: list[str], **kwargs: Any) -> int:
        return 0

    monkeypatch.setattr("raglite._insert.insert_documents", fake_insert)
    monkeypatch.setattr("raglite._delete.delete_documents", fake_delete)
    monkeypatch.setattr("raglite._incremental._existing_documents_by_key", lambda *args: {})
    cache = get_retrieval_cache()
    cache.set(cache.make_key("q", num_results=3), (["c1"], [1.0]))
    document = Document(id="doc-new", filename="new.md", content="content")
    incremental_module.sync_documents([document], config=test_config)
    assert cache.get(cache.make_key("q", num_results=3)) is None
