"""Low-confidence secondary retrieval.

When the top rerank score of the first retrieval pass is below a confidence threshold, retrieve
again with an expanded candidate pool and keep the best results of both passes by rerank score.
This is inspired by the FAQ iterative retrieval of WeKnora, where each iteration starts at twice
the already-retrieved depth. The confidence signal is the rerank score; without a reranker the
secondary pass is skipped, since there is no signal to threshold on.
"""

from raglite._config import RAGLiteConfig
from raglite._database import ChunkSpan
from raglite._typing import BasicSearchMethod, ChunkId, MetadataFilter


def search_with_secondary_retrieval(  # noqa: PLR0913
    query: str,
    *,
    num_results: int = 8,
    oversample: int = 4,
    search: BasicSearchMethod | None = None,
    metadata_filter: MetadataFilter | None = None,
    config: RAGLiteConfig | None = None,
) -> tuple[list[ChunkId], list[float]]:
    """Search and rerank with a low-confidence secondary retrieval pass.

    Parameters
    ----------
    query
        The query to search for.
    num_results
        The number of results to return.
    oversample
        The oversampling factor applied to the initial candidate pool.
    search
        The underlying search method. Defaults to the config's search method.
    metadata_filter
        An optional metadata filter.
    config
        The RAGLite config. `secondary_retrieval` enables or disables the secondary pass,
        `rerank_confidence_threshold` is the confidence threshold on the top rerank score, and
        `secondary_retrieval_multiplier` scales the expanded candidate pool.

    Returns
    -------
    tuple
        The top chunk ids and their rerank scores (or search scores without a reranker).
    """
    from raglite._search import hybrid_search, rerank_chunks_with_scores

    config = config or RAGLiteConfig()
    search = search or hybrid_search
    # First pass: search an oversampled candidate pool and rerank it.
    chunk_ids, _ = search(
        query, num_results=oversample * num_results, metadata_filter=metadata_filter, config=config
    )
    reranked_chunks, rerank_scores = rerank_chunks_with_scores(query, chunk_ids, config=config)
    ranked: dict[str, float] = {
        chunk.id: (rerank_scores[i] if rerank_scores else 0.0)
        for i, chunk in enumerate(reranked_chunks)
    }
    top_scores = list(ranked.values())[:num_results]
    # Skip the secondary pass when disabled, or when there is no confidence signal (no reranker).
    if not config.secondary_retrieval or rerank_scores is None:
        final_ids = list(ranked)[:num_results]
        return final_ids, [ranked[chunk_id] for chunk_id in final_ids]
    # Second pass: if the top rerank score is below the confidence threshold, retrieve again with
    # an expanded candidate pool, rerank the new candidates, and keep the best of both passes.
    threshold = config.rerank_confidence_threshold
    if top_scores and max(top_scores) < threshold:
        multiplier = config.secondary_retrieval_multiplier
        expanded_ids, _ = search(
            query,
            num_results=oversample * num_results * multiplier,
            metadata_filter=metadata_filter,
            config=config,
        )
        new_ids = [chunk_id for chunk_id in expanded_ids if chunk_id not in ranked]
        if new_ids:
            new_chunks, new_scores = rerank_chunks_with_scores(query, new_ids, config=config)
            for i, chunk in enumerate(new_chunks):
                ranked[chunk.id] = new_scores[i] if new_scores else 0.0
    # Rank all candidates (both passes) by rerank score and keep the top results.
    ranked_pairs = sorted(ranked.items(), key=lambda item: item[1], reverse=True)[:num_results]
    return [chunk_id for chunk_id, _ in ranked_pairs], [score for _, score in ranked_pairs]


def secondary_retrieval_search(
    query: str,
    *,
    num_results: int = 8,
    metadata_filter: MetadataFilter | None = None,
    config: RAGLiteConfig | None = None,
) -> tuple[list[ChunkId], list[float]]:
    """Search with low-confidence secondary retrieval. Usable as a config search method."""
    return search_with_secondary_retrieval(
        query, num_results=num_results, metadata_filter=metadata_filter, config=config
    )


def search_and_rerank_chunk_spans_with_secondary_retrieval(  # noqa: PLR0913
    query: str,
    *,
    num_results: int = 8,
    oversample: int = 4,
    neighbors: tuple[int, ...] | None = (-1, 1),
    search: BasicSearchMethod | None = None,
    metadata_filter: MetadataFilter | None = None,
    config: RAGLiteConfig | None = None,
) -> list[ChunkSpan]:
    """Search with low-confidence secondary retrieval and collate the results into chunk spans."""
    from raglite._search import retrieve_chunk_spans

    chunk_ids, _ = search_with_secondary_retrieval(
        query,
        num_results=num_results,
        oversample=oversample,
        search=search,
        metadata_filter=metadata_filter,
        config=config,
    )
    return retrieve_chunk_spans(chunk_ids, neighbors=neighbors, config=config)
