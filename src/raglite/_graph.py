"""Graph-enhanced retrieval: entity/relation extraction and a third RRF ranking.

This module adds a graph path to RAGLite's hybrid search. The storage follows LightRAG rather than
WeKnora — the graph lives in the same DuckDB/PostgreSQL database as the chunks, so there is no
Neo4j, no extra service, and no new dependency — while the weighting, the chunk-level projection,
and the query-time expansion follow WeKnora's `graph.go` and `search_entity.go`.

The pipeline has two halves.

Index side (`build_graph`)
    Every chunk that has not been graph-extracted yet goes through an entity/relation extractor
    once, then the extracted entities, their chunk links, and their relations are aggregated into
    the graph tables. Entities are deduplicated by normalised title, so the same entity mentioned
    by different chunks or documents collapses onto one node whose chunk links span the corpus.
    Because a document's chunk ids are derived from its content hash, an unchanged document keeps
    its chunk ids and is therefore skipped — which is what gives the graph layer the same
    incremental behaviour as the vector layer.

    Two WeKnora mechanisms then run over the result.

    1. `recompute_relation_weights` scores every edge by pointwise mutual information and by the
       strength the extractor stated, and scales the combination into the 1-10 range. PMI is what
       makes an edge meaningful: two entities that merely both occur often are weakly related even
       if they co-occur a lot, while two rarer entities that always appear together are strongly
       related.
    2. `build_chunk_relations` projects the entity graph onto the chunks: two chunks that share an
       entity are linked, with the relation's weight and the endpoints' combined degree. This is
       the projection that lets retrieval reach a chunk that never mentions the query entity.

Query side (`graph_search`)
    The question is matched against the stored entity titles (lexically, so no model call, unless
    `GraphConfig.query_extraction` asks the extractor for the question's entities), the matched
    entities are expanded over the relations, and the chunks linked to the resulting weighted
    entity set are scored. Those chunks are then expanded over the chunk graph, so a chunk reached
    through a related entity can pull in the chunk that entity shares with it. The ranking is fed
    into `reciprocal_rank_fusion`, which is deliberately where WeKnora also merges its graph
    results: before reranking, not after.

    `graph_search` accepts `exclude_chunk_ids` for WeKnora's `filterSeenChunk` behaviour: the
    chunks the other paths already found can be dropped so the graph path only contributes new
    ones. It is off by default, because RRF already rewards agreement between paths.

Both halves degrade rather than fail: an extractor error costs one chunk's extraction and is
reported in the build report, a graph ranking with no matches is simply empty, and an empty
ranking contributes nothing to RRF — so a query whose entities are not in the graph behaves
exactly like the two-path baseline.

The path is off by default (`GraphConfig.enabled`). Every chunk costs one extra LLM call at index
time and the ranking only adds value for relational or corpus-level questions, so it should be
turned on only after benchmarking it against the two-path baseline on a real question set.
"""

from __future__ import annotations

import math
import re
import warnings
from collections import defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel
from pydantic import Field as PydanticField
from sqlalchemy import delete as sa_delete
from sqlmodel import Session, col, func, select

from raglite._config import DEFAULT_GRAPH_EXTRACTION_PROMPT, GraphConfig, RAGLiteConfig
from raglite._database import (
    Chunk,
    GraphChunk,
    GraphChunkRelation,
    GraphEntity,
    GraphEntityChunk,
    GraphRelation,
    create_database_engine,
    hash_bytes,
)
from raglite._metadata_filter import build_metadata_filter_condition
from raglite._typing import ChunkId, DocumentId, GraphEntityId, MetadataFilter

# The default number of entities and relations requested per chunk.
GRAPH_BATCH_SIZE = 256
# A query that matches more seed entities than this is almost certainly matching junk.
MAX_SEED_ENTITY_CANDIDATES = 64
# Upper bound on the candidate chunks carried into the existence and metadata-filter query, so a
# broad query against a large graph cannot build an unbounded `IN` clause.
GRAPH_CANDIDATE_POOL = 512
# The scale of the extractor's stated strength, which WeKnora asks for on a 1-10 scale.
MAX_RELATION_STRENGTH = 10.0


class GraphError(Exception):
    """A graph extraction or retrieval error."""


# ---------------------------------------------------------------------------
# Extraction records.
# ---------------------------------------------------------------------------


class ExtractedEntity(BaseModel):
    """An entity extracted from a chunk of text."""

    title: str = PydanticField(
        ..., description="The entity's canonical name exactly as written in the text."
    )
    type: str = PydanticField(
        default="other", description="A short category, e.g. person, organization, concept."
    )
    description: str = PydanticField(
        default="", description="One sentence about the entity, grounded in the text."
    )


class ExtractedRelation(BaseModel):
    """A relation between two entities extracted from a chunk of text."""

    source: str = PydanticField(..., description="The title of the source entity.")
    target: str = PydanticField(..., description="The title of the target entity.")
    description: str = PydanticField(
        default="", description="How the source entity relates to the target entity."
    )
    strength: float = PydanticField(
        default=1.0,
        description=(
            "How strongly the text states this relation, from 1 (mentioned in passing) to 10 (the "
            "text is about this relation)."
        ),
    )

    def normalized_strength(self) -> float:
        """Return the stated strength on a 0-1 scale, clamped to the 1-10 range."""
        return min(max(self.strength, 1.0), MAX_RELATION_STRENGTH) / MAX_RELATION_STRENGTH


class GraphExtraction(BaseModel):
    """The entities and relations extracted from a single chunk of text."""

    entities: list[ExtractedEntity] = PydanticField(
        default_factory=list, description="The entities mentioned in the text."
    )
    relations: list[ExtractedRelation] = PydanticField(
        default_factory=list, description="The relations stated between those entities."
    )
    # The system prompt is a class variable so that `extract_with_llm` can pick it up without it
    # leaking into the JSON schema. Extractors override it with their configured prompt.
    system_prompt: ClassVar[str] = ""


# ---------------------------------------------------------------------------
# Entity/relation extractors.
# ---------------------------------------------------------------------------


@runtime_checkable
class EntityExtractor(Protocol):
    """Protocol for an entity and relation extractor.

    Implementations must be safe to call from multiple threads, because `build_graph` extracts
    chunks in parallel.
    """

    def extract(self, text: str) -> GraphExtraction:
        """Extract the entities and relations stated in `text`."""
        ...  # pragma: no cover


def build_extraction_prompt(graph_config: GraphConfig) -> str:
    """Build the extraction prompt, appending WeKnora's `ExtractConfig` constraints.

    WeKnora injects the permitted entity types, relation types, and worked examples into the
    extraction prompt as few-shot constraints. RAGLite does the same when they are configured, and
    leaves the prompt untouched when they are not: an over-constrained schema silently drops the
    entities a personal corpus is actually about.
    """
    lines = [graph_config.extraction_prompt]
    if graph_config.entity_types:
        lines.append("Entity types to use: " + ", ".join(graph_config.entity_types) + ".")
    if graph_config.relation_types:
        lines.append("Relation types to use: " + ", ".join(graph_config.relation_types) + ".")
    if graph_config.extraction_examples.strip():
        lines.append("Examples:\n" + graph_config.extraction_examples.strip())
    return "\n".join(lines)


@dataclass
class LLMEntityExtractor:
    """An entity/relation extractor backed by an LLM through LiteLLM.

    The model comes from the `RAGLiteConfig` it is given, so a single config controls both
    generation and graph extraction unless `GraphConfig.extractor` names another model.
    """

    config: RAGLiteConfig
    prompt: str = DEFAULT_GRAPH_EXTRACTION_PROMPT
    temperature: float = 0.1
    max_tokens: int = 1024

    def extract(self, text: str) -> GraphExtraction:
        """Extract the entities and relations stated in `text`."""
        from raglite._extract import extract_with_llm

        # `extract_with_llm` reads the system prompt off the return type, so build a model whose
        # `system_prompt` is the configured one.
        response_type: type[GraphExtraction] = type(
            "GraphExtractionResponse",
            (GraphExtraction,),
            {"system_prompt": self.prompt, "__module__": __name__},
        )
        kwargs: dict[str, Any] = {"temperature": self.temperature}
        if self.max_tokens > 0:
            kwargs["max_tokens"] = self.max_tokens
        return extract_with_llm(response_type, text, config=self.config, **kwargs)


# Candidate generators for the offline extractor. Latin capitalised runs and quoted spans are
# matched directly; CJK has no segmenter available, so it is mined as closed n-grams instead (see
# `_closed_cjk_ngrams`).
_LATIN_TERM_PATTERN = re.compile(r"[A-Z][A-Za-z0-9]*(?:[ _\-][A-Z][A-Za-z0-9]*)*")
_QUOTED_TERM_PATTERN = re.compile(
    r"「[^」\n]{1,40}」|《[^》\n]{1,40}》|\"[^\"\n]{1,40}\"|“[^”\n]{1,40}”"
)
_CJK_RUN_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
CJK_NGRAM_MIN = 2
CJK_NGRAM_MAX = 4

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。！？；;])\s+|\n+")

# Function words that show up as capitalised words at the start of English sentences.
_RULE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "but",
        "by",
        "for",
        "from",
        "however",
        "if",
        "in",
        "it",
        "its",
        "moreover",
        "note",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "then",
        "therefore",
        "these",
        "this",
        "those",
        "thus",
        "to",
        "we",
        "when",
        "which",
        "while",
        "with",
    }
)


def _closed_cjk_ngrams(text: str, min_occurrences: int) -> dict[str, int]:
    """Mine repeated closed CJK n-grams as a segmentation-free stand-in for term extraction.

    A gram is kept only when no longer kept gram accounts for all of its occurrences, because then
    the longer gram carries strictly more information: given "量子纠错" on every occurrence of
    "量子", keeping "量子" adds nothing. This is a crude statistical term extractor, not a
    linguistic one — it is what makes an offline Chinese graph build possible at all.
    """
    counts: defaultdict[str, int] = defaultdict(int)
    for run in _CJK_RUN_PATTERN.findall(text):
        max_n = min(CJK_NGRAM_MAX, len(run))
        for n in range(CJK_NGRAM_MIN, max_n + 1):
            for i in range(len(run) - n + 1):
                counts[run[i : i + n]] += 1
    repeated = {gram: count for gram, count in counts.items() if count >= min_occurrences}
    kept: dict[str, int] = {}
    for gram in sorted(repeated, key=lambda g: (-len(g), g)):
        if any(gram in longer and repeated[longer] >= repeated[gram] for longer in kept):
            continue
        kept[gram] = repeated[gram]
    return kept


@dataclass
class CooccurrenceExtractor:
    """A deterministic, dependency-free extractor for offline graph builds and ablations.

    It keeps repeated Latin terms, quoted spans, and repeated closed CJK n-grams, and relates the
    terms that co-occur in a sentence. It is explicitly **not** a semantic extractor: it finds no
    relation the text does not imply through proximity, and it mislabels some terms. It exists so
    that the graph path can be built, tested, and ablated without a model endpoint, and so that an
    A/B comparison of the retrieval fusion is reproducible offline. Use `LLMEntityExtractor` for
    real graph quality.
    """

    min_occurrences: int = 2
    max_entities_per_chunk: int = 16
    max_relations_per_chunk: int = 128

    def extract(self, text: str) -> GraphExtraction:
        """Extract candidate terms and the relations their sentence co-occurrence implies."""
        sentences = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
        counts: defaultdict[str, int] = defaultdict(int)  # case-folded key -> occurrences
        surface: dict[str, str] = {}  # case-folded key -> preferred surface form
        always_keep: set[str] = set()

        def record(title: str) -> None:
            key = title.casefold()
            surface.setdefault(key, title)
            counts[key] += 1

        for sentence in sentences:
            for candidate in _LATIN_TERM_PATTERN.findall(sentence):
                title = normalize_entity_title(candidate)
                if title and title.casefold() not in _RULE_STOPWORDS:
                    record(title)
            for candidate in _QUOTED_TERM_PATTERN.findall(sentence):
                title = normalize_entity_title(candidate)
                if title:
                    record(title)
                    always_keep.add(title.casefold())
        for ngram, count in _closed_cjk_ngrams(text, self.min_occurrences).items():
            key = ngram.casefold()
            surface.setdefault(key, ngram)
            counts[key] = max(counts[key], count)
        # Keep repeated terms (and every quoted span), preferring frequent and long ones.
        kept = sorted(
            (
                (key, count)
                for key, count in counts.items()
                if count >= self.min_occurrences or key in always_keep
            ),
            key=lambda item: (-item[1], -len(item[0]), item[0]),
        )[: self.max_entities_per_chunk]
        entities = [ExtractedEntity(title=surface[key], type="term") for key, _ in kept]
        # Relate the kept terms that share a sentence.
        relations: list[ExtractedRelation] = []
        for sentence in sentences:
            folded = sentence.casefold()
            present = [key for key, _ in kept if key in folded]
            for i, source in enumerate(present):
                for target in present[i + 1 :]:
                    if len(relations) >= self.max_relations_per_chunk:
                        break
                    relations.append(
                        ExtractedRelation(
                            source=surface[source],
                            target=surface[target],
                            description="co-occurs in the same sentence",
                            strength=1.0,
                        )
                    )
        return GraphExtraction(entities=entities, relations=relations)


def resolve_entity_extractor(graph_config: GraphConfig, config: RAGLiteConfig) -> EntityExtractor:
    """Resolve the configured extractor into an `EntityExtractor`.

    `None` builds an LLM extractor on the RAGLite LLM, a string is used as the LiteLLM model
    identifier of an LLM extractor, and an `EntityExtractor` instance is used as-is. This mirrors
    `resolve_ocr_engine` / `resolve_vlm_engine` in the multimodal processor.
    """
    extractor = graph_config.extractor
    if extractor is None:
        return LLMEntityExtractor(
            config,
            prompt=build_extraction_prompt(graph_config),
            temperature=graph_config.extraction_temperature,
            max_tokens=graph_config.extraction_max_tokens,
        )
    if isinstance(extractor, str):
        return LLMEntityExtractor(
            replace(config, llm=extractor),
            prompt=build_extraction_prompt(graph_config),
            temperature=graph_config.extraction_temperature,
            max_tokens=graph_config.extraction_max_tokens,
        )
    return extractor


# ---------------------------------------------------------------------------
# Title normalisation and matching.
# ---------------------------------------------------------------------------


_EDGE_PUNCTUATION = " \t\r\n\"'“”‘’「」『』《》〈〉()[]{}<>,.;:!?、。，；：！？·—–-"
_WHITESPACE_PATTERN = re.compile(r"\s+")
_LATIN_ONLY_PATTERN = re.compile(r"[a-z0-9][a-z0-9 \-.,'&/_]*")


def normalize_entity_title(title: str) -> str:
    """Normalise an entity title for deduplication and query matching."""
    normalized = title.strip().strip(_EDGE_PUNCTUATION).strip()
    return _WHITESPACE_PATTERN.sub(" ", normalized)


def entity_id_for_title(title: str) -> GraphEntityId:
    """Derive a stable entity id from a normalised, case-folded title."""
    return hash_bytes(normalize_entity_title(title).casefold().encode())


def _is_latin(title: str) -> bool:
    """Whether the title is written entirely in the Latin script."""
    return _LATIN_ONLY_PATTERN.fullmatch(title.casefold()) is not None


def title_in_query(title: str, query: str) -> bool:
    """Whether an entity title is mentioned in the query.

    Latin titles must appear on word boundaries so that, say, "art" does not match "partial". CJK
    titles are matched as substrings, which is the right granularity for unsegmented text.
    """
    lower_title = title.casefold()
    lower_query = query.casefold()
    if not lower_title:
        return False
    if _is_latin(lower_title):
        pattern = rf"(?<![a-z0-9]){re.escape(lower_title)}(?![a-z0-9])"
        return re.search(pattern, lower_query) is not None
    return lower_title in lower_query


# ---------------------------------------------------------------------------
# Graph construction.
# ---------------------------------------------------------------------------


def _batched(values: Sequence[Any], size: int = GRAPH_BATCH_SIZE) -> Iterator[list[Any]]:
    """Yield `values` in fixed-size batches."""
    for i in range(0, len(values), size):
        yield list(values[i : i + size])


def _rowcount(result: Any) -> int:
    """Read a DELETE's row count, treating the drivers that report -1 as unknown."""
    rowcount = getattr(result, "rowcount", None)
    if rowcount is None or rowcount < 0:
        return 0
    return int(rowcount)


def _extract_one(extractor: EntityExtractor, text: str) -> GraphExtraction | None:
    """Run one extraction, returning None instead of raising so a chunk can degrade."""
    try:
        return extractor.extract(text)
    except Exception as error:  # noqa: BLE001
        warnings.warn(f"Graph extraction failed for a chunk: {error}", stacklevel=2)
        return None


def prune_graph(*, config: RAGLiteConfig | None = None) -> dict[str, int]:
    """Remove graph rows that no longer have a chunk behind them.

    Chunk ids are content hashes, so a document that changed or was deleted leaves graph rows
    pointing at chunks that no longer exist. This is the graph counterpart of the vector layer's
    incremental delete.
    """
    config = config or RAGLiteConfig()
    removed = {"chunk_links": 0, "chunk_relations": 0, "chunks": 0, "entities": 0, "relations": 0}
    with Session(create_database_engine(config)) as session:
        # Drop the links of chunks that are gone, then the bookkeeping rows of those same chunks.
        removed["chunk_links"] = _rowcount(
            session.execute(
                sa_delete(GraphEntityChunk).where(
                    ~col(GraphEntityChunk.chunk_id).in_(select(Chunk.id))
                )
            )
        )
        # Drop the projected chunk-chunk edges whose endpoints are gone, on either side.
        removed["chunk_relations"] = _rowcount(
            session.execute(
                sa_delete(GraphChunkRelation).where(
                    ~col(GraphChunkRelation.source_chunk_id).in_(select(Chunk.id))
                )
            )
        ) + _rowcount(
            session.execute(
                sa_delete(GraphChunkRelation).where(
                    ~col(GraphChunkRelation.target_chunk_id).in_(select(Chunk.id))
                )
            )
        )
        removed["chunks"] = _rowcount(
            session.execute(
                sa_delete(GraphChunk).where(~col(GraphChunk.chunk_id).in_(select(Chunk.id)))
            )
        )
        # Drop entities that lost all their chunk links.
        removed["entities"] = _rowcount(
            session.execute(
                sa_delete(GraphEntity).where(
                    ~col(GraphEntity.id).in_(select(GraphEntityChunk.entity_id))
                )
            )
        )
        # Drop relations whose endpoints no longer exist.
        removed["relations"] = _rowcount(
            session.execute(
                sa_delete(GraphRelation).where(
                    ~col(GraphRelation.source_id).in_(select(GraphEntity.id))
                )
            )
        )
        session.commit()
    return removed


def clear_graph(*, config: RAGLiteConfig | None = None) -> None:
    """Delete the whole graph (entities, relations, links, projection, and bookkeeping)."""
    config = config or RAGLiteConfig()
    with Session(create_database_engine(config)) as session:
        for table in (
            GraphChunkRelation,
            GraphEntityChunk,
            GraphRelation,
            GraphEntity,
            GraphChunk,
        ):
            session.execute(sa_delete(table))
        session.commit()


def pending_graph_chunks(
    *,
    document_ids: Sequence[DocumentId] | None = None,
    config: RAGLiteConfig | None = None,
) -> list[ChunkId]:
    """List the chunks that have not been graph-extracted yet."""
    config = config or RAGLiteConfig()
    with Session(create_database_engine(config)) as session:
        done = set(session.exec(select(GraphChunk.chunk_id)).all())
        statement = select(Chunk.id)
        if document_ids is not None:
            statement = statement.where(col(Chunk.document_id).in_(list(document_ids)))
        return [chunk_id for chunk_id in session.exec(statement).all() if chunk_id not in done]


def build_graph(
    *,
    document_ids: Sequence[DocumentId] | None = None,
    max_workers: int | None = None,
    prune_missing: bool = True,
    config: RAGLiteConfig | None = None,
) -> dict[str, Any]:
    """Extract entities and relations for the chunks that are not in the graph yet.

    Only the delta is extracted: chunks already recorded in `graph_chunk` are skipped, and because
    a document's chunk ids are content hashes, an unchanged document is skipped entirely. That is
    the graph layer's counterpart of `sync_documents`'s content-hash incremental indexing.

    Parameters
    ----------
    document_ids
        Restrict the build to these documents. `None` (the default) covers every document.
    max_workers
        The number of extraction worker threads. Defaults to `GraphConfig.extraction_max_workers`.
    prune_missing
        Remove graph rows whose chunks no longer exist before extracting. Leave this on unless you
        are building a graph across documents that are being written concurrently.
    config
        The RAGLite config to use.

    Returns
    -------
    dict
        A report with the number of extracted, skipped, and failed chunks, plus the total number
        of entities and relations in the graph after the build.
    """
    config = config or RAGLiteConfig()
    graph_config = config.graph
    if prune_missing:
        prune_graph(config=config)
    extractor = resolve_entity_extractor(graph_config, config)
    workers = max_workers or graph_config.extraction_max_workers
    with Session(create_database_engine(config)) as session:
        done = set(session.exec(select(GraphChunk.chunk_id)).all())
        statement = select(Chunk.id, Chunk.document_id, Chunk.headings, Chunk.body)
        if document_ids is not None:
            statement = statement.where(col(Chunk.document_id).in_(list(document_ids)))
        rows: list[tuple[ChunkId, DocumentId, str, str]] = [
            (chunk_id, document_id, headings, body)
            for chunk_id, document_id, headings, body in session.exec(statement).all()
        ]
    pending = [row for row in rows if row[0] not in done]
    report: dict[str, Any] = {
        "chunks_extracted": 0,
        "chunks_skipped": len(rows) - len(pending),
        "chunks_failed": 0,
    }
    if pending:
        # Extract in parallel (the extractor is required to be thread-safe; LLM extraction is I/O
        # bound, which is exactly what threads are good at here).
        texts = [f"{headings}\n{body}" if headings else body for _, _, headings, body in pending]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            extractions = list(pool.map(lambda text: _extract_one(extractor, text), texts))
        report["chunks_failed"] = sum(1 for extraction in extractions if extraction is None)
        _write_graph([(row[0], row[1]) for row in pending], extractions, config=config)
        report["chunks_extracted"] = len(pending) - report["chunks_failed"]
    # WeKnora scores the edges and projects them onto the chunks after extraction: both depend on
    # corpus-wide statistics, so they run over the whole graph rather than over the delta.
    report["relations_scored"] = recompute_relation_weights(config=config)
    if graph_config.build_chunk_graph:
        report["chunk_relations"] = build_chunk_relations(config=config)
    report.update(graph_stats(config=config))
    return report


def _write_graph(
    rows: Sequence[tuple[ChunkId, DocumentId]],
    extractions: Sequence[GraphExtraction | None],
    *,
    config: RAGLiteConfig,
) -> None:
    """Aggregate the extractions and write them to the graph tables."""
    # Aggregate the extractions in memory first: dedupe entities by normalised title, accumulate
    # their chunk links, and record the relations between the surviving entities.
    entity_titles: dict[GraphEntityId, str] = {}
    entity_types: dict[GraphEntityId, str] = {}
    entity_descriptions: dict[GraphEntityId, str] = {}
    entity_chunks: defaultdict[GraphEntityId, set[ChunkId]] = defaultdict(set)
    relation_descriptions: dict[tuple[GraphEntityId, GraphEntityId], str] = {}
    relation_observations: defaultdict[tuple[GraphEntityId, GraphEntityId], int] = defaultdict(int)
    relation_strengths: defaultdict[tuple[GraphEntityId, GraphEntityId], float] = defaultdict(float)
    succeeded: list[tuple[ChunkId, DocumentId]] = []
    for (chunk_id, document_id), extraction in zip(rows, extractions, strict=True):
        if extraction is None:
            continue
        succeeded.append((chunk_id, document_id))
        local: dict[str, GraphEntityId] = {}
        for entity in extraction.entities:
            title = normalize_entity_title(entity.title)
            if not title or len(title) > config.graph.max_entity_title_length:
                continue
            entity_id = entity_id_for_title(title)
            local[title.casefold()] = entity_id
            entity_titles.setdefault(entity_id, title)
            entity_types.setdefault(entity_id, entity.type or "other")
            if entity.description and not entity_descriptions.get(entity_id):
                entity_descriptions[entity_id] = entity.description
            entity_chunks[entity_id].add(chunk_id)
        for relation in extraction.relations:
            source = local.get(normalize_entity_title(relation.source).casefold())
            target = local.get(normalize_entity_title(relation.target).casefold())
            if source is None or target is None or source == target:
                continue
            key = (source, target)
            if relation.description and not relation_descriptions.get(key):
                relation_descriptions[key] = relation.description
            relation_observations[key] += 1
            relation_strengths[key] += relation.normalized_strength()
    if not succeeded:
        return
    with Session(create_database_engine(config)) as session:
        # Record the chunks that yielded nothing too, so they are not extracted on every build.
        if not entity_titles:
            for chunk_id, document_id in succeeded:
                session.add(GraphChunk(chunk_id=chunk_id, document_id=document_id))
            session.commit()
            return
        entity_ids = list(entity_titles)
        existing_entities: dict[str, GraphEntity] = {}
        for batch in _batched(entity_ids):
            entity_records: list[GraphEntity] = list(
                session.exec(select(GraphEntity).where(col(GraphEntity.id).in_(batch))).all()
            )
            for entity_record in entity_records:
                existing_entities[entity_record.id] = entity_record
        # Upsert entities: never overwrite a description we already have.
        for entity_id in entity_ids:
            stored = existing_entities.get(entity_id)
            if stored is None:
                session.add(
                    GraphEntity(
                        id=entity_id,
                        title=entity_titles[entity_id],
                        entity_type=entity_types.get(entity_id, "other"),
                        description=entity_descriptions.get(entity_id, ""),
                    )
                )
                continue
            if not stored.description and entity_descriptions.get(entity_id):
                stored.description = entity_descriptions[entity_id]
                session.add(stored)
        # Which (entity, chunk) links already exist for the chunks we just processed.
        new_chunk_ids = [chunk_id for chunk_id, _ in succeeded]
        existing_links: set[tuple[str, str]] = set()
        for batch in _batched(new_chunk_ids):
            stored_links: list[GraphEntityChunk] = list(
                session.exec(
                    select(GraphEntityChunk).where(col(GraphEntityChunk.chunk_id).in_(batch))
                ).all()
            )
            for stored_link in stored_links:
                existing_links.add((stored_link.entity_id, stored_link.chunk_id))
        for entity_id, linked_chunk_ids in entity_chunks.items():
            for chunk_id in linked_chunk_ids:
                if (entity_id, chunk_id) not in existing_links:
                    session.add(GraphEntityChunk(entity_id=entity_id, chunk_id=chunk_id))
        # Upsert relations: the primary key is the (source, target) pair, so repeated mentions
        # accumulate into one weighted edge.
        existing_relations: dict[tuple[str, str], GraphRelation] = {}
        for batch in _batched(entity_ids):
            stored_relations: list[GraphRelation] = list(
                session.exec(
                    select(GraphRelation).where(
                        col(GraphRelation.source_id).in_(batch),
                        col(GraphRelation.target_id).in_(batch),
                    )
                ).all()
            )
            for stored_relation in stored_relations:
                existing_relations[(stored_relation.source_id, stored_relation.target_id)] = (
                    stored_relation
                )
        for relation_key, observations in relation_observations.items():
            stored_edge = existing_relations.get(relation_key)
            strength = relation_strengths[relation_key] / observations
            if stored_edge is None:
                session.add(
                    GraphRelation(
                        source_id=relation_key[0],
                        target_id=relation_key[1],
                        description=relation_descriptions.get(relation_key, ""),
                        weight=1.0,  # `recompute_relation_weights` scores it below.
                        strength=strength,
                        observations=observations,
                    )
                )
                continue
            # The stored strength is the mean over every chunk that stated the relation, so an
            # incremental build that sees one more mention cannot drag it around.
            total = stored_edge.observations + observations
            stored_edge.strength = (
                stored_edge.strength * stored_edge.observations + relation_strengths[relation_key]
            ) / total
            stored_edge.observations = total
            if not stored_edge.description and relation_descriptions.get(relation_key):
                stored_edge.description = relation_descriptions[relation_key]
            session.add(stored_edge)
        # Bookkeeping last, so an interrupted write leaves the chunks pending rather than lost.
        for chunk_id, document_id in succeeded:
            session.add(GraphChunk(chunk_id=chunk_id, document_id=document_id))
        session.commit()


# ---------------------------------------------------------------------------
# Edge weighting and the chunk-level projection (WeKnora's graph.go).
# ---------------------------------------------------------------------------


def _entity_supports(session: Session) -> tuple[dict[GraphEntityId, int], int]:
    """Return the number of chunks each entity occurs in, and the number of indexed chunks."""
    rows = session.exec(
        select(GraphEntityChunk.entity_id, func.count(col(GraphEntityChunk.chunk_id))).group_by(
            col(GraphEntityChunk.entity_id)
        )
    ).all()
    supports = {entity_id: int(count) for entity_id, count in rows}
    total_chunks = session.exec(
        select(func.count(func.distinct(col(GraphEntityChunk.chunk_id))))
    ).one()
    return supports, int(total_chunks)


def _relation_cooccurrence_supports(
    session: Session, pairs: set[tuple[GraphEntityId, GraphEntityId]]
) -> dict[tuple[GraphEntityId, GraphEntityId], int]:
    """Return, for each entity pair, the number of chunks that mention both endpoints.

    Counting per pair would mean one query per relation, so the links are folded into
    `chunk -> entities` once and only the pairs that are actually relations are counted.
    """
    if not pairs:
        return {}
    normalized_pairs = {(source, target) for source, target in pairs}
    normalized_pairs |= {(target, source) for source, target in pairs}
    chunk_entities: defaultdict[ChunkId, list[GraphEntityId]] = defaultdict(list)
    for entity_id, chunk_id in session.exec(
        select(GraphEntityChunk.entity_id, GraphEntityChunk.chunk_id)
    ).all():
        chunk_entities[chunk_id].append(entity_id)
    # An entity that occurs in every chunk contributes nothing: it co-occurs with everything.
    supports: defaultdict[tuple[GraphEntityId, GraphEntityId], int] = defaultdict(int)
    for entity_ids in chunk_entities.values():
        if len(entity_ids) < 2:  # noqa: PLR2004
            continue
        for i, source in enumerate(entity_ids):
            for target in entity_ids[i + 1 :]:
                if (source, target) in normalized_pairs:
                    supports[(source, target)] += 1
    return dict(supports)


def pointwise_mutual_information(
    joint_probability: float, source_probability: float, target_probability: float
) -> float:
    """Return WeKnora's PMI: `log2(P(x, y) / (P(x) * P(y)))`, clamped at zero.

    A positive value says the two entities co-occur more than chance; zero says they co-occur no
    more than independent entities would.
    """
    if joint_probability <= 0 or source_probability <= 0 or target_probability <= 0:
        return 0.0
    return max(math.log2(joint_probability / (source_probability * target_probability)), 0.0)


def recompute_relation_weights(*, config: RAGLiteConfig | None = None) -> int:
    """Score every relation by PMI and strength, as WeKnora's `graph.go` does.

    The weight is `1 + weight_scale * (pmi_weight * normalised PMI + strength_weight * normalised
    strength)`, which lands in the 1-10 range: `pmi_weight` and `strength_weight` are 0.6 and 0.4
    in WeKnora, and `weight_scale` is 9. PMI is normalised by the strongest PMI in the graph, so
    the scale is comparable across corpora.

    Returns the number of relations that were scored.
    """
    config = config or RAGLiteConfig()
    graph_config = config.graph
    with Session(create_database_engine(config)) as session:
        relations = list(session.exec(select(GraphRelation)).all())
        if not relations:
            return 0
        supports, total_chunks = _entity_supports(session)
        cooccurrence_supports = _relation_cooccurrence_supports(
            session, {(relation.source_id, relation.target_id) for relation in relations}
        )
        # PMI first: it depends on corpus-wide statistics, so every edge is scored before any is
        # normalised.
        pmis: dict[tuple[GraphEntityId, GraphEntityId], float] = {}
        for relation in relations:
            key = (relation.source_id, relation.target_id)
            joint = cooccurrence_supports.get(key, 0) / total_chunks if total_chunks else 0.0
            source = supports.get(relation.source_id, 0) / total_chunks if total_chunks else 0.0
            target = supports.get(relation.target_id, 0) / total_chunks if total_chunks else 0.0
            pmis[key] = pointwise_mutual_information(joint, source, target)
        max_pmi = max(pmis.values(), default=0.0)
        for relation in relations:
            key = (relation.source_id, relation.target_id)
            normalized_pmi = pmis[key] / max_pmi if max_pmi > 0 else 0.0
            normalized_strength = min(max(relation.strength, 0.0), 1.0)
            combined = (
                graph_config.pmi_weight * normalized_pmi
                + graph_config.strength_weight * normalized_strength
            )
            relation.weight = 1.0 + graph_config.weight_scale * combined
            session.add(relation)
        session.commit()
    return len(relations)


def build_chunk_relations(*, config: RAGLiteConfig | None = None) -> int:
    """Project the entity graph onto the chunks, as WeKnora's `buildChunkGraph` does.

    Two chunks that share an entity are linked in both directions with the weight of the relation
    that connected them and the combined degree of the endpoint entities. Retrieval can then walk
    from a chunk that mentions the query entity to a chunk that only mentions a related one.

    The projection is rebuilt in full rather than merged, because an edge's weight depends on the
    whole graph: a partial update would either double count or drift away from the entity graph.

    Returns the number of chunk-chunk edges written.
    """
    config = config or RAGLiteConfig()
    graph_config = config.graph
    with Session(create_database_engine(config)) as session:
        session.execute(sa_delete(GraphChunkRelation))
        relations = list(session.exec(select(GraphRelation)).all())
        if not relations:
            session.commit()
            return 0
        entity_ids = {
            entity_id
            for relation in relations
            for entity_id in (relation.source_id, relation.target_id)
        }
        chunks_by_entity: defaultdict[GraphEntityId, list[ChunkId]] = defaultdict(list)
        for batch in _batched(list(entity_ids)):
            for entity_id, chunk_id in session.exec(
                select(GraphEntityChunk.entity_id, GraphEntityChunk.chunk_id).where(
                    col(GraphEntityChunk.entity_id).in_(batch)
                )
            ).all():
                chunks_by_entity[entity_id].append(chunk_id)
        edges: dict[tuple[ChunkId, ChunkId], tuple[float, int]] = {}
        for relation in relations:
            source_chunks = chunks_by_entity.get(relation.source_id, [])
            target_chunks = chunks_by_entity.get(relation.target_id, [])
            degree = len(source_chunks) + len(target_chunks)
            for source_chunk_id in source_chunks[: graph_config.max_chunks_per_entity]:
                for target_chunk_id in target_chunks[: graph_config.max_chunks_per_entity]:
                    if source_chunk_id == target_chunk_id:
                        continue
                    for key in (
                        (source_chunk_id, target_chunk_id),
                        (target_chunk_id, source_chunk_id),
                    ):
                        weight, stored_degree = edges.get(key, (0.0, 0))
                        edges[key] = (weight + relation.weight, max(stored_degree, degree))
        for (source_chunk_id, target_chunk_id), (weight, degree) in edges.items():
            session.add(
                GraphChunkRelation(
                    source_chunk_id=source_chunk_id,
                    target_chunk_id=target_chunk_id,
                    weight=weight,
                    degree=degree,
                )
            )
        session.commit()
    return len(edges)


# ---------------------------------------------------------------------------
# Graph retrieval.
# ---------------------------------------------------------------------------


def match_seed_entities(
    query: str, entities: Sequence[tuple[GraphEntityId, str]], graph_config: GraphConfig
) -> list[GraphEntityId]:
    """Match the query against the stored entity titles, without a model call.

    Longer titles win, because a longer match is more specific: a query mentioning "quantum error
    correction" should seed that entity rather than a bare "quantum".
    """
    matched: list[tuple[int, GraphEntityId]] = []
    for entity_id, title in entities:
        normalized = normalize_entity_title(title)
        if not normalized or len(normalized) > graph_config.max_entity_title_length:
            continue
        if title_in_query(normalized, query):
            matched.append((len(normalized), entity_id))
    matched.sort(key=lambda item: -item[0])
    return [entity_id for _, entity_id in matched[:MAX_SEED_ENTITY_CANDIDATES]]


def _live_chunk_ids(
    session: Session,
    chunk_ids: Sequence[ChunkId],
    metadata_filter: MetadataFilter | None,
    *,
    dialect: str,
) -> list[ChunkId]:
    """Keep the chunk ids that still exist and survive the metadata filter, preserving order.

    The existence check matters because the graph declares no foreign keys, so deleting a document
    can leave graph links pointing at chunks that are gone. Order is preserved, because the graph
    ranking is carried by the order of this list.
    """
    if not chunk_ids:
        return []
    condition = build_metadata_filter_condition(Chunk.metadata_, metadata_filter, dialect=dialect)
    statement = select(Chunk.id).where(col(Chunk.id).in_(list(chunk_ids)))
    if condition is not None:
        statement = statement.where(condition)
    allowed = set(session.exec(statement).all())
    return [chunk_id for chunk_id in chunk_ids if chunk_id in allowed]


def extract_query_entities(
    query: str,
    entities: Sequence[tuple[GraphEntityId, str]],
    *,
    config: RAGLiteConfig,
) -> list[GraphEntityId]:
    """Extract the query's entities with the configured extractor and match them by title.

    This is WeKnora's `PluginExtractEntity`: the question, not the corpus, is what decides which
    entities seed the search. It costs one extractor call per query, which is why
    `GraphConfig.query_extraction` is off by default and the lexical matcher covers the common
    case. An extractor failure degrades to no additional seeds rather than failing the query.
    """
    graph_config = config.graph
    try:
        extraction = resolve_entity_extractor(graph_config, config).extract(query)
    except Exception as error:  # noqa: BLE001
        warnings.warn(f"Query entity extraction failed: {error}", stacklevel=2)
        return []
    id_by_title = {
        normalize_entity_title(title).casefold(): entity_id for entity_id, title in entities
    }
    seed_ids: list[GraphEntityId] = []
    for entity in extraction.entities:
        entity_id = id_by_title.get(normalize_entity_title(entity.title).casefold())
        if entity_id is not None and entity_id not in seed_ids:
            seed_ids.append(entity_id)
    return seed_ids[: graph_config.seed_entity_limit]


def _chunk_neighbors(
    session: Session, chunk_ids: Sequence[ChunkId]
) -> dict[ChunkId, list[tuple[ChunkId, float]]]:
    """Return the chunk-graph neighbours of `chunk_ids` with their normalised edge weights."""
    neighbors: defaultdict[ChunkId, list[tuple[ChunkId, float]]] = defaultdict(list)
    if not chunk_ids:
        return {}
    for batch in _batched(list(chunk_ids)):
        edges = session.exec(
            select(
                GraphChunkRelation.source_chunk_id,
                GraphChunkRelation.target_chunk_id,
                GraphChunkRelation.weight,
            ).where(col(GraphChunkRelation.source_chunk_id).in_(batch))
        ).all()
        for source_chunk_id, target_chunk_id, weight in edges:
            # Edges carry the sum of the relation weights that produced them, so a neighbour's
            # pull is the edge's share of the strongest possible edge.
            neighbors[source_chunk_id].append((target_chunk_id, min(weight / 10.0, 1.0)))
    return dict(neighbors)


def _expand_chunk_neighbors(
    chunk_scores: dict[ChunkId, float],
    session: Session,
    graph_config: GraphConfig,
) -> dict[ChunkId, float]:
    """Expand the scored chunks over the chunk graph, decaying the score per hop.

    This is what the projection buys: a chunk that only mentions a related entity is reachable
    even though no seed entity points at it. Scores are merged with `max`, so a chunk that is
    both directly seeded and reachable is never demoted by the decayed path.
    """
    scores = dict(chunk_scores)
    frontier = list(scores)
    for hop in range(max(0, graph_config.chunk_neighbor_hops)):
        if not frontier:
            break
        neighbors = _chunk_neighbors(session, frontier)
        decay = graph_config.chunk_neighbor_weight ** (hop + 1)
        next_frontier: list[ChunkId] = []
        for chunk_id in frontier:
            for neighbor_id, edge_weight in neighbors.get(chunk_id, []):
                candidate = scores[chunk_id] * decay * edge_weight
                if candidate <= 0:
                    continue
                if candidate > scores.get(neighbor_id, 0.0):
                    scores[neighbor_id] = candidate
                    next_frontier.append(neighbor_id)
        frontier = next_frontier
    return scores


def graph_search(  # noqa: C901
    query: str,
    *,
    num_results: int = 3,
    metadata_filter: MetadataFilter | None = None,
    exclude_chunk_ids: Sequence[ChunkId] | None = None,
    config: RAGLiteConfig | None = None,
) -> tuple[list[ChunkId], list[float]]:
    """Rank chunks by their graph proximity to the entities mentioned in the query.

    The query is matched against the stored entity titles — lexically by default, and with the
    extractor as well when `GraphConfig.query_extraction` is set — the matched entities are
    expanded over the relations for `neighbor_hops` hops, and the chunks linked to the resulting
    weighted entity set are scored. Those chunks are then expanded over the chunk graph for
    `chunk_neighbor_hops` hops. The ranking is meant to be fused with the vector and keyword
    rankings by `reciprocal_rank_fusion`, before reranking.

    Parameters
    ----------
    query
        The question to rank chunks for.
    num_results
        The maximum number of chunks to return.
    metadata_filter
        An optional metadata filter the returned chunks must satisfy.
    exclude_chunk_ids
        Chunks to drop from the ranking, WeKnora's `filterSeenChunk`: the chunks the vector and
        keyword paths already found, so the graph only contributes what they missed.
    config
        The RAGLite config to use.

    Returns
    -------
    tuple
        The ranked chunk ids and their scores. Both are empty when the graph is disabled or when
        the query matches no entity, so the caller can treat it as a no-op.
    """
    config = config or RAGLiteConfig()
    graph_config = config.graph
    if not graph_config.enabled:
        return [], []
    with Session(create_database_engine(config)) as session:
        dialect = session.get_bind().dialect.name
        entities = list(session.exec(select(GraphEntity.id, GraphEntity.title)).all())
        if not entities:
            return [], []
        seed_ids = match_seed_entities(query, entities, graph_config)
        if graph_config.query_extraction:
            seed_ids = list(
                dict.fromkeys(seed_ids + extract_query_entities(query, entities, config=config))
            )
        if not seed_ids:
            return [], []
        seed_ids = seed_ids[: graph_config.seed_entity_limit]
        # Expand the seed set over the relations, decaying the weight per hop.
        entity_weights: dict[GraphEntityId, float] = dict.fromkeys(seed_ids, 1.0)
        frontier = list(seed_ids)
        for hop in range(max(0, graph_config.neighbor_hops)):
            if not frontier:
                break
            relations: list[GraphRelation] = []
            for batch in _batched(frontier):
                relations.extend(
                    session.exec(
                        select(GraphRelation).where(col(GraphRelation.source_id).in_(batch))
                    ).all()
                )
            next_frontier: list[GraphEntityId] = []
            decay = graph_config.neighbor_weight ** (hop + 1)
            for relation in relations:
                if relation.target_id in entity_weights or decay <= 0:
                    continue
                entity_weights[relation.target_id] = decay
                next_frontier.append(relation.target_id)
            frontier = next_frontier
        # Collect and score the chunks linked to the weighted entity set.
        chunk_scores: defaultdict[ChunkId, float] = defaultdict(float)
        for batch in _batched(list(entity_weights)):
            links = session.exec(
                select(GraphEntityChunk.entity_id, GraphEntityChunk.chunk_id).where(
                    col(GraphEntityChunk.entity_id).in_(batch)
                )
            ).all()
            for entity_id, chunk_id in links:
                chunk_scores[chunk_id] += entity_weights[entity_id]
        if not chunk_scores:
            return [], []
        # Walk the chunk graph, then drop what the other paths already found.
        scores = (
            _expand_chunk_neighbors(dict(chunk_scores), session, graph_config)
            if graph_config.chunk_neighbor_hops
            else dict(chunk_scores)
        )
        excluded = set(exclude_chunk_ids or ())
        ordered = [
            chunk_id
            for chunk_id in sorted(scores, key=lambda c: (-scores[c], c))
            if chunk_id not in excluded
        ]
        ordered = _live_chunk_ids(
            session, ordered[:GRAPH_CANDIDATE_POOL], metadata_filter, dialect=dialect
        )[:num_results]
    return ordered, [scores[chunk_id] for chunk_id in ordered]


@dataclass(frozen=True)
class GraphEntityView:
    """An entity as returned by `query_knowledge_graph`."""

    title: str
    entity_type: str
    description: str
    chunk_ids: tuple[ChunkId, ...] = ()


@dataclass(frozen=True)
class GraphRelationView:
    """A relation as returned by `query_knowledge_graph`."""

    source: str
    target: str
    description: str
    weight: float


@dataclass(frozen=True)
class GraphQueryResult:
    """A subgraph around the entities mentioned in a query.

    This is RAGLite's counterpart of WeKnora's `query_knowledge_graph` agent tool: the entities
    the query matched, the relations between them and their neighbours, and the chunks those
    entities were extracted from. It is meant for an agent that wants to explore the graph, or for
    a frontend that wants to draw it, not for the retrieval path itself.
    """

    query: str
    entities: tuple[GraphEntityView, ...] = ()
    relations: tuple[GraphRelationView, ...] = ()
    chunk_ids: tuple[ChunkId, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def query_knowledge_graph(  # noqa: C901
    query: str,
    *,
    max_entities: int = 8,
    neighbor_hops: int = 1,
    config: RAGLiteConfig | None = None,
) -> GraphQueryResult:
    """Return the subgraph around the entities mentioned in `query`.

    The query is matched against the stored entity titles lexically (and with the extractor when
    `GraphConfig.query_extraction` is set), the matched entities are expanded for `neighbor_hops`
    hops, and the entities, the relations among them, and the chunks they were extracted from are
    returned. An unmatched query yields an empty result rather than an error.
    """
    config = config or RAGLiteConfig()
    graph_config = config.graph
    with Session(create_database_engine(config)) as session:
        rows = list(session.exec(select(GraphEntity)).all())
        if not rows:
            return GraphQueryResult(query=query)
        entities = [(entity.id, entity.title) for entity in rows]
        seed_ids = match_seed_entities(query, entities, graph_config)
        if graph_config.query_extraction:
            seed_ids = list(
                dict.fromkeys(seed_ids + extract_query_entities(query, entities, config=config))
            )
        seed_ids = seed_ids[:max_entities]
        if not seed_ids:
            return GraphQueryResult(query=query)
        entity_weights: dict[GraphEntityId, float] = dict.fromkeys(seed_ids, 1.0)
        frontier = list(seed_ids)
        for hop in range(max(0, neighbor_hops)):
            if not frontier:
                break
            relations: list[GraphRelation] = []
            for batch in _batched(frontier):
                relations.extend(
                    session.exec(
                        select(GraphRelation).where(col(GraphRelation.source_id).in_(batch))
                    ).all()
                )
            decay = graph_config.neighbor_weight ** (hop + 1)
            next_frontier: list[GraphEntityId] = []
            for relation in relations:
                if relation.target_id in entity_weights or decay <= 0:
                    continue
                entity_weights[relation.target_id] = decay
                next_frontier.append(relation.target_id)
            frontier = next_frontier
        selected = list(entity_weights)
        by_id: dict[GraphEntityId, GraphEntity] = {
            entity.id: entity for entity in rows if entity.id in entity_weights
        }
        chunks_by_entity: defaultdict[GraphEntityId, list[ChunkId]] = defaultdict(list)
        for batch in _batched(selected):
            for entity_id, chunk_id in session.exec(
                select(GraphEntityChunk.entity_id, GraphEntityChunk.chunk_id).where(
                    col(GraphEntityChunk.entity_id).in_(batch)
                )
            ).all():
                chunks_by_entity[entity_id].append(chunk_id)
        stored_relations = [
            relation
            for relation in session.exec(select(GraphRelation)).all()
            if relation.source_id in entity_weights and relation.target_id in entity_weights
        ]
        title_by_id = {entity.id: entity.title for entity in rows}
        entity_views = tuple(
            GraphEntityView(
                title=by_id[entity_id].title,
                entity_type=by_id[entity_id].entity_type,
                description=by_id[entity_id].description,
                chunk_ids=tuple(sorted(chunks_by_entity.get(entity_id, []))),
            )
            for entity_id in selected
            if entity_id in by_id
        )
        relation_views = tuple(
            GraphRelationView(
                source=title_by_id.get(relation.source_id, ""),
                target=title_by_id.get(relation.target_id, ""),
                description=relation.description,
                weight=relation.weight,
            )
            for relation in stored_relations
        )
        chunk_ids = tuple(
            dict.fromkeys(
                chunk_id
                for entity_id in selected
                for chunk_id in chunks_by_entity.get(entity_id, [])
            )
        )
    return GraphQueryResult(
        query=query,
        entities=entity_views,
        relations=relation_views,
        chunk_ids=chunk_ids,
        metadata={"seeds": len(seed_ids), "entities": len(entity_views)},
    )


def graph_stats(*, config: RAGLiteConfig | None = None) -> dict[str, int]:
    """Report the size of the graph.

    `chunks_indexed` counts the chunks that have been through extraction (including the ones that
    yielded nothing), which is the graph layer's counterpart of the vector layer's indexed chunks.
    """
    config = config or RAGLiteConfig()
    with Session(create_database_engine(config)) as session:
        return {
            "entities": session.exec(select(func.count(col(GraphEntity.id)))).one(),
            "relations": session.exec(select(func.count(col(GraphRelation.source_id)))).one(),
            "entity_chunk_links": session.exec(
                select(func.count(col(GraphEntityChunk.chunk_id)))
            ).one(),
            "chunk_relations": session.exec(
                select(func.count(col(GraphChunkRelation.source_chunk_id)))
            ).one(),
            "chunks_indexed": session.exec(select(func.count(col(GraphChunk.chunk_id)))).one(),
        }
