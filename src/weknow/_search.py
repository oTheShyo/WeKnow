"""Search and retrieve chunks."""

import contextlib
import logging
import re
import string
from collections import defaultdict
from itertools import groupby
from typing import Any, ClassVar

import numpy as np
from langdetect import LangDetectException, detect
from pydantic import BaseModel, Field, create_model
from sqlalchemy.orm import joinedload
from sqlmodel import Session, and_, col, func, or_, select, text

from weknow._config import RAGLiteConfig
from weknow._database import (
    Chunk,
    ChunkEmbedding,
    ChunkSpan,
    IndexMetadata,
    _adapt_metadata,
    create_database_engine,
)
from weknow._embed import embed_strings
from weknow._extract import extract_with_llm
from weknow._graph import graph_search
from weknow._insert import _get_database_metadata
from weknow._metadata_filter import build_metadata_filter_condition, build_metadata_filter_sql
from weknow._typing import BasicSearchMethod, ChunkId, FloatVector, MetadataFilter, MetadataValue

logger = logging.getLogger(__name__)

# L1 cache of query embeddings, keyed by (embedder config, query). Query embedding is the most
# expensive per-query step before the database lookup, and repeated queries (pagination, retries,
# cache misses at the retrieval level) hit it over and over.
QUERY_EMBEDDING_CACHE_MAX_SIZE = 128
_query_embedding_cache: dict[tuple[str, str], FloatVector] = {}


def _embed_query_cached(query: str, config: RAGLiteConfig) -> FloatVector:
    """Embed a query string, caching the result per (embedder config, query)."""
    cache_key = (str(config.embedder), query)
    cached = _query_embedding_cache.get(cache_key)
    if cached is not None:
        return cached
    query_embedding = embed_strings([query], config=config)[0, :]
    if len(_query_embedding_cache) >= QUERY_EMBEDDING_CACHE_MAX_SIZE:
        _query_embedding_cache.pop(next(iter(_query_embedding_cache)))
    _query_embedding_cache[cache_key] = query_embedding
    return query_embedding


def vector_search(
    query: str | FloatVector,
    *,
    num_results: int = 3,
    oversample: int = 4,
    metadata_filter: MetadataFilter | None = None,
    config: RAGLiteConfig | None = None,
) -> tuple[list[ChunkId], list[float]]:
    """Search chunks using ANN vector search."""
    # Read the config.
    config = config or RAGLiteConfig()
    # Normalize metadata filter values to lists.
    metadata_filter = _adapt_metadata(metadata_filter)
    # If self_query is enabled, extract metadata filters from the query.
    if config.self_query and isinstance(query, str):
        self_query_filter = _self_query(query, config=config)
        metadata_filter = {**self_query_filter, **(metadata_filter or {})}
    # Embed the query.
    query_embedding = (
        _embed_query_cached(query, config) if isinstance(query, str) else np.ravel(query)
    )
    # Apply the query adapter to the query embedding.
    if (
        config.vector_search_query_adapter
        and (Q := IndexMetadata.get("default", config=config).get("query_adapter"))  # noqa: N806
        is not None
    ):
        query_embedding = (Q @ query_embedding).astype(query_embedding.dtype)
    # Rank the chunks by relevance according to the L∞ norm of the similarities of the multi-vector
    # chunk embeddings to the query embedding with a single query.
    with Session(create_database_engine(config)) as session:
        corrected_oversample = oversample * config.chunk_max_size / RAGLiteConfig.chunk_max_size
        num_hits = round(corrected_oversample) * max(num_results, 10)

        dist = ChunkEmbedding.embedding.distance(  # type: ignore[attr-defined]
            query_embedding, metric=config.vector_search_distance_metric
        ).label("dist")
        sim = (1.0 - dist).label("sim")

        # Build the query that retrieves the top chunk embeddings.
        if not metadata_filter:
            # No metadata filter: use an index to get the top results by distance.
            top_vectors = (
                select(ChunkEmbedding.chunk_id, sim, dist).order_by(dist).limit(num_hits).subquery()
            )
        else:

            def _apply_metadata_filter(query_builder: Any) -> Any:
                condition = build_metadata_filter_condition(
                    Chunk.metadata_,
                    metadata_filter,
                    dialect=session.get_bind().dialect.name,
                )
                return query_builder.where(condition) if condition is not None else query_builder

            # Count how many results match the given metadata filter.
            metadata_count_query = _apply_metadata_filter(
                select(func.count(col(ChunkEmbedding.chunk_id))).join(
                    Chunk,
                    ChunkEmbedding.chunk_id == Chunk.id,  # type: ignore[arg-type]
                )
            )
            metadata_count = session.exec(metadata_count_query).one()

            if metadata_count <= 100_000:  # noqa: PLR2004
                # Metadata filter produces few results: filter first, then order by distance.
                filtered_chunks_subquery = _apply_metadata_filter(
                    select(ChunkEmbedding.chunk_id).join(
                        Chunk,
                        ChunkEmbedding.chunk_id == Chunk.id,  # type: ignore[arg-type]
                    )
                )
                top_vectors = (
                    select(ChunkEmbedding.chunk_id, sim, dist)
                    .where(col(ChunkEmbedding.chunk_id).in_(filtered_chunks_subquery))
                    .order_by(dist)
                    .limit(num_hits)
                    .subquery()
                )
            else:
                # Metadata filter produces many results: first order by distance, then filter.
                top_by_distance = (
                    select(ChunkEmbedding.chunk_id, sim, dist)
                    .order_by(dist)
                    .limit(1_000_000)
                    .subquery()
                )
                top_vectors = (
                    _apply_metadata_filter(
                        select(
                            top_by_distance.c.chunk_id,
                            top_by_distance.c.sim,
                            top_by_distance.c.dist,
                        ).select_from(
                            top_by_distance.join(Chunk, top_by_distance.c.chunk_id == Chunk.id)
                        )
                    )
                    .order_by(top_by_distance.c.dist)
                    .limit(num_hits)
                    .subquery()
                )

        sim_norm = func.max(top_vectors.c.sim).label("sim_norm")
        statement = (
            select(top_vectors.c.chunk_id, sim_norm)
            .group_by(top_vectors.c.chunk_id)
            .order_by(sim_norm.desc())
            .limit(num_results)
        )
        rows = session.exec(statement).all()
        chunk_ids = [row[0] for row in rows]
        similarity = [float(row[1]) for row in rows]
    return chunk_ids, similarity


def keyword_search(
    query: str,
    *,
    num_results: int = 3,
    metadata_filter: MetadataFilter | None = None,
    config: RAGLiteConfig | None = None,
) -> tuple[list[ChunkId], list[float]]:
    """Search chunks using BM25 keyword search."""
    # Read the config.
    config = config or RAGLiteConfig()
    # Normalize metadata filter values to lists.
    metadata_filter = _adapt_metadata(metadata_filter)
    # If self_query is enabled, extract metadata filters from the query.
    if config.self_query and isinstance(query, str):
        self_query_filter = _self_query(query, config=config)
        metadata_filter = {**self_query_filter, **(metadata_filter or {})}
    # Connect to the database.
    with Session(create_database_engine(config)) as session:
        dialect = session.get_bind().dialect.name

        if dialect == "postgresql":
            # Convert the query to a tsquery [1].
            # [1] https://www.postgresql.org/docs/current/textsearch-controls.html
            query_escaped = re.sub(f"[{re.escape(string.punctuation)}]", " ", query)
            tsv_query = " | ".join(query_escaped.split())

            base_sql = """
                SELECT id as chunk_id,
                       ts_rank(to_tsvector('simple', body), to_tsquery('simple', :tsv_query)) as score
                FROM chunk
                WHERE to_tsvector('simple', body) @@ to_tsquery('simple', :tsv_query)
            """

            params = {"tsv_query": tsv_query, "limit": num_results}
            if metadata_filter:
                metadata_filter_sql, metadata_filter_params = build_metadata_filter_sql(
                    metadata_filter, dialect="postgresql"
                )
                base_sql += metadata_filter_sql
                params.update(metadata_filter_params)

            base_sql += """
                ORDER BY score DESC
                LIMIT :limit;
            """

            statement = text(base_sql)
            results = session.execute(statement, params)

        elif dialect == "duckdb":
            base_sql = """
                SELECT chunk_id, score
                FROM (
                    SELECT id AS chunk_id, fts_main_chunk.match_bm25(id, :query) AS score
                    FROM chunk
                    WHERE 1=1
            """

            params = {"query": query, "limit": num_results}
            if metadata_filter:
                metadata_filter_sql, metadata_filter_params = build_metadata_filter_sql(
                    metadata_filter, dialect="duckdb"
                )
                base_sql += metadata_filter_sql
                params.update(metadata_filter_params)

            base_sql += """
                ) sq
                WHERE score IS NOT NULL
                ORDER BY score DESC
                LIMIT :limit;
            """

            statement = text(base_sql)
            results = session.execute(statement, params)
        # Unpack the results.
        results = list(results)  # type: ignore[assignment]
        chunk_ids = [result.chunk_id for result in results]
        keyword_score = [result.score for result in results]
    return chunk_ids, keyword_score


def reciprocal_rank_fusion(
    rankings: list[list[ChunkId]], *, k: int = 60, weights: list[float] | None = None
) -> tuple[list[ChunkId], list[float]]:
    """Reciprocal Rank Fusion."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        error = "The number of weights must match the number of rankings."
        raise ValueError(error)
    # Compute the RRF score.
    chunk_id_score: defaultdict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights, strict=True):
        for i, chunk_id in enumerate(ranking):
            chunk_id_score[chunk_id] += weight / (k + i)
    # Exit early if there are no results to fuse.
    if not chunk_id_score:
        return [], []
    # Rank RRF results according to descending RRF score.
    rrf_chunk_ids, rrf_score = zip(
        *sorted(chunk_id_score.items(), key=lambda x: x[1], reverse=True), strict=True
    )
    return list(rrf_chunk_ids), list(rrf_score)


def hybrid_search(  # noqa: PLR0913
    query: str,
    *,
    num_results: int = 3,
    oversample: int = 2,
    vector_search_weight: float | None = None,
    keyword_search_weight: float | None = None,
    graph_search_weight: float | None = None,
    metadata_filter: MetadataFilter | None = None,
    config: RAGLiteConfig | None = None,
) -> tuple[list[ChunkId], list[float]]:
    """Search chunks by combining ANN vector search with BM25 keyword search and a graph ranking.

    The RRF weights default to the config values (`hybrid_vector_weight` /
    `hybrid_keyword_weight` / `hybrid_graph_weight`) so that the weights are configurable instead
    of hardcoded.

    The graph ranking is only fused in when `config.graph.enabled` is set, and because an empty
    ranking contributes nothing to RRF, a query that matches no graph entity behaves exactly like
    the two-path baseline. As in WeKnora, the graph ranking is merged *before* reranking.
    """
    # Read the config and resolve the weights.
    config = config or RAGLiteConfig()
    vector_search_weight = (
        config.hybrid_vector_weight if vector_search_weight is None else vector_search_weight
    )
    keyword_search_weight = (
        config.hybrid_keyword_weight if keyword_search_weight is None else keyword_search_weight
    )
    graph_search_weight = (
        config.hybrid_graph_weight if graph_search_weight is None else graph_search_weight
    )
    # Run the vector and keyword searches.
    vs_chunk_ids, _ = vector_search(
        query, num_results=oversample * num_results, metadata_filter=metadata_filter, config=config
    )
    ks_chunk_ids, _ = keyword_search(
        query, num_results=oversample * num_results, metadata_filter=metadata_filter, config=config
    )
    rankings = [vs_chunk_ids, ks_chunk_ids]
    weights = [vector_search_weight, keyword_search_weight]
    # Add the graph ranking as a third path when the graph extension is enabled.
    if config.graph.enabled:
        # `deduplicate_seen_chunks` is WeKnora's `filterSeenChunk`: the graph then only contributes
        # chunks the vector and keyword paths missed. It is off by default, because RRF already
        # rewards a chunk that several paths agree on.
        seen_chunk_ids = (
            list(dict.fromkeys(vs_chunk_ids + ks_chunk_ids))
            if config.graph.deduplicate_seen_chunks
            else None
        )
        gs_chunk_ids, _ = graph_search(
            query,
            num_results=oversample * num_results,
            metadata_filter=metadata_filter,
            exclude_chunk_ids=seen_chunk_ids,
            config=config,
        )
        rankings.append(gs_chunk_ids)
        weights.append(graph_search_weight)
    # Combine the results with Reciprocal Rank Fusion (RRF).
    chunk_ids, hybrid_score = reciprocal_rank_fusion(rankings, weights=weights)
    chunk_ids, hybrid_score = chunk_ids[:num_results], hybrid_score[:num_results]
    return chunk_ids, hybrid_score


def retrieve_chunks(
    chunk_ids: list[ChunkId], *, config: RAGLiteConfig | None = None
) -> list[Chunk]:
    """Retrieve chunks by their ids."""
    if not chunk_ids:
        return []
    with Session(create_database_engine(config := config or RAGLiteConfig())) as session:
        chunks = list(
            session.exec(
                select(Chunk)
                .where(col(Chunk.id).in_(chunk_ids))
                # Eagerly load chunk.document.
                .options(joinedload(Chunk.document))  # type: ignore[arg-type]
            ).all()
        )
    chunks = sorted(chunks, key=lambda chunk: chunk_ids.index(chunk.id))
    return chunks


def retrieve_chunk_spans(
    chunk_ids: list[ChunkId] | list[Chunk],
    *,
    neighbors: tuple[int, ...] | None = (-1, 1),
    config: RAGLiteConfig | None = None,
) -> list[ChunkSpan]:
    """Group chunks into contiguous chunk spans and retrieve them.

    Chunk spans are ordered according to the aggregate relevance of their underlying chunks, as
    determined by the order in which they are provided to this function.
    """
    # Exit early if the input is empty.
    if not chunk_ids:
        return []
    # Retrieve the chunks.
    config = config or RAGLiteConfig()
    chunks: list[Chunk] = (
        retrieve_chunks(chunk_ids, config=config)  # type: ignore[arg-type,assignment]
        if all(isinstance(chunk_id, ChunkId) for chunk_id in chunk_ids)
        else chunk_ids
    )
    # Assign a reciprocal ranking score to each chunk based on its position in the original list.
    chunk_id_to_score = {chunk.id: 1 / (i + 1) for i, chunk in enumerate(chunks)}
    # Extend the chunks with their neighbouring chunks.
    with Session(create_database_engine(config)) as session:
        if neighbors:
            neighbor_conditions = [
                and_(Chunk.document_id == chunk.document_id, Chunk.index == chunk.index + offset)
                for chunk in chunks
                for offset in neighbors
            ]
            chunks += list(
                session.exec(
                    select(Chunk)
                    .where(or_(*neighbor_conditions))
                    # Eagerly load chunk.document.
                    .options(joinedload(Chunk.document))  # type: ignore[arg-type]
                ).all()
            )
    # Deduplicate and sort the chunks by document_id and index (needed for groupby).
    unique_chunks = sorted(set(chunks), key=lambda chunk: (chunk.document_id, chunk.index))
    # Group the chunks into contiguous segments.
    chunk_spans: list[ChunkSpan] = []
    for _, group in groupby(unique_chunks, key=lambda chunk: chunk.document_id):
        chunk_sequence: list[Chunk] = []
        for chunk in group:
            if not chunk_sequence or chunk.index == chunk_sequence[-1].index + 1:
                chunk_sequence.append(chunk)
            else:
                chunk_spans.append(ChunkSpan(chunks=chunk_sequence))
                chunk_sequence = [chunk]
        chunk_spans.append(ChunkSpan(chunks=chunk_sequence))
    # Rank segments according to the aggregate relevance of their chunks.
    chunk_spans.sort(
        key=lambda chunk_span: sum(
            chunk_id_to_score.get(chunk.id, 0.0) for chunk in chunk_span.chunks
        ),
        reverse=True,
    )
    return chunk_spans


def rerank_chunks_with_scores(
    query: str, chunk_ids: list[ChunkId] | list[Chunk], *, config: RAGLiteConfig | None = None
) -> tuple[list[Chunk], list[float] | None]:
    """Rerank chunks and return them with their relevance scores.

    Returns the chunks in retrieval order with scores None when no reranker is configured, or when
    reranking fails (degrade to retrieval order instead of failing the whole search).
    """
    # Retrieve the chunks.
    config = config or RAGLiteConfig()
    chunks: list[Chunk] = (
        retrieve_chunks(chunk_ids, config=config)  # type: ignore[arg-type,assignment]
        if all(isinstance(chunk_id, ChunkId) for chunk_id in chunk_ids)
        else chunk_ids
    )
    # Exit early if no reranker is configured or if the input is empty.
    if not config.reranker or not chunks:
        return chunks, None
    # Select the reranker.
    if isinstance(config.reranker, dict):
        # Detect the languages of the chunks and queries.
        with contextlib.suppress(LangDetectException):
            langs = {detect(str(chunk)) for chunk in chunks}
            langs.add(detect(query))
        # If all chunks and the query are in the same language, use a language-specific reranker.
        rerankers = config.reranker
        if len(langs) == 1 and (lang := next(iter(langs))) in rerankers:
            reranker = rerankers[lang]
        else:
            reranker = rerankers.get("other")
    else:
        # A specific reranker was configured.
        reranker = config.reranker
    # Rerank the chunks, degrading to retrieval order on failure.
    if reranker:
        try:
            results = reranker.rank(query=query, docs=[str(chunk) for chunk in chunks])
        except Exception:
            logger.warning("Reranking failed; falling back to retrieval order.", exc_info=True)
        else:
            reranked_chunks = [chunks[result.doc_id] for result in results.results]
            rerank_scores = [float(result.score) for result in results.results]
            return reranked_chunks, rerank_scores
    return chunks, None


def rerank_chunks(
    query: str, chunk_ids: list[ChunkId] | list[Chunk], *, config: RAGLiteConfig | None = None
) -> list[Chunk]:
    """Rerank chunks according to their relevance to a given query."""
    chunks, _ = rerank_chunks_with_scores(query, chunk_ids, config=config)
    return chunks


def search_and_rerank_chunks(  # noqa: PLR0913
    query: str,
    *,
    num_results: int = 8,
    oversample: int = 4,
    search: BasicSearchMethod = hybrid_search,
    config: RAGLiteConfig | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> list[Chunk]:
    """Search and rerank chunks."""
    chunk_ids, _ = search(
        query, num_results=oversample * num_results, metadata_filter=metadata_filter, config=config
    )
    chunks = rerank_chunks(query, chunk_ids, config=config)[:num_results]
    return chunks


def search_and_rerank_chunk_spans(  # noqa: PLR0913
    query: str,
    *,
    num_results: int = 8,
    oversample: int = 4,
    neighbors: tuple[int, ...] | None = (-1, 1),
    search: BasicSearchMethod = hybrid_search,
    config: RAGLiteConfig | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> list[ChunkSpan]:
    """Search and rerank chunks, and then collate into chunk spans."""
    chunk_ids, _ = search(
        query, num_results=oversample * num_results, metadata_filter=metadata_filter, config=config
    )
    chunks = rerank_chunks(query, chunk_ids, config=config)[:num_results]
    chunk_spans = retrieve_chunk_spans(chunks, neighbors=neighbors, config=config)
    return chunk_spans


SELF_QUERY_PROMPT = """
You are an expert assistant that extracts metadata filters from user queries to help search a knowledge base.

Instructions:
1. For each metadata field, populate it only if the query can be reasonably mapped to one or more allowed values.
2. If a field clearly matches multiple allowed values, return all corresponding numeric IDs for that field.
3. If the query is general, ambiguous, or does not clearly map to any allowed value for a field, return None for that field.
4. For each populated field, return only the numeric ID(s) defined in the allowed options. Do not return text labels. Do not infer or invent IDs.
5. Output your answer as a JSON object with field names as keys and lists of IDs or None as values.

Examples:

Allowed options:
- category: {0: "Technology", 1: "Health", 2: "Finance"}
- region: {0: "Europe", 1: "Asia", 2: "Americas"}

Query: "Show me the latest news in Technology from Asia and Europe."
Reasoning: The query explicitly mentions "Technology", which matches category ID 0. It also explicitly mentions "Asia" (region ID 1) and "Europe" (region ID 0). Both fields have clear matches.
Output: {"category": [0], "region": [1, 0]}

Query: "Show me Health articles."
Reasoning: The query explicitly mentions "Health", which matches category ID 1. The query does not mention any region, so region is None.
Output: {"category": [1], "region": null}

Query: "What is the price of a Bugatti Chiron?"
Reasoning: The query does not mention any category ("Technology", "Health", or "Finance") or any region ("Europe", "Asia", or "Americas"). No fields match.
Output: {"category": null, "region": null}
""".strip()


def _self_query(
    query: str,
    *,
    system_prompt: str = SELF_QUERY_PROMPT,
    config: RAGLiteConfig | None = None,
) -> MetadataFilter:
    """Extract metadata filters from a natural language query."""
    config = config or RAGLiteConfig()
    # Retrieve the available metadata from the database.
    metadata_records = _get_database_metadata(config=config)
    if not metadata_records:
        return {}
    # Create dynamic Pydantic model for the metadata filter
    field_ids_mapping: dict[str, dict[int, MetadataValue]] = {}
    field_definitions: dict[str, Any] = {}
    field_definitions["system_prompt"] = (ClassVar[str], system_prompt)
    # Note:
    # The LLM tends to return escaped Unicode or hexadecimal strings when asked to output
    # labels directly. By assigning each allowed metadata value a numeric ID and asking
    # the model to return only IDs, we avoid encoding issues and reliably map results
    # back to their actual metadata values afterward.
    for record in metadata_records:
        field_ids_mapping[record.name] = dict(enumerate(record.values))
        # Store the mapping in
        description = (
            "Return ONLY IDs from this set (use IDs, not labels). "
            f"Allowed options: {field_ids_mapping[record.name]}"
        )
        field_definitions[record.name] = (
            list[int] | None,
            Field(default=None, description=description),
        )
    metadata_filter_model = create_model(
        "MetadataFilterModel", **field_definitions, __base__=BaseModel
    )
    # Call extract_with_llm
    try:
        result = extract_with_llm(
            return_type=metadata_filter_model,
            user_prompt=query,
            config=config,
            temperature=0.0,  # Deterministic output if the model allows
            drop_params=True,
        )
    except ValueError as e:
        logger.debug("Failed to extract metadata filter: %s", e)
        return {}
    else:
        # Convert the extracted metadata filter from IDs back to actual metadata values.
        metadata_filter_by_id = result.model_dump(exclude_none=True)
        metadata_filter: dict[str, list[MetadataValue] | MetadataValue] = {}
        for field, value_ids in metadata_filter_by_id.items():
            value_mapping = field_ids_mapping.get(field, {})
            metadata_values = list(
                dict.fromkeys(
                    value_mapping[value_id]
                    for value_id in value_ids
                    if value_id in value_mapping  # handle potential out-of-range IDs gracefully
                )
            )
            if metadata_values:
                metadata_filter[field] = metadata_values
        return metadata_filter
