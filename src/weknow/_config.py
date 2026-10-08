"""RAGLite config."""

import contextlib
import os
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from platformdirs import user_data_dir
from sqlalchemy.engine import URL

from weknow._lazy_llama import llama_supports_gpu_offload
from weknow._typing import ChunkId, MetadataFilter, SearchMethod

if TYPE_CHECKING:
    from weknow._graph import EntityExtractor
    from weknow._multimodal import OCREngine, VLMEngine

# Suppress rerankers output on import until [1] is fixed.
# [1] https://github.com/AnswerDotAI/rerankers/issues/36
with contextlib.redirect_stdout(StringIO()):
    from rerankers.models.flashrank_ranker import FlashRankRanker
    from rerankers.models.ranker import BaseRanker


cache_path = Path(user_data_dir("weknow", ensure_exists=True))


DEFAULT_IMAGE_TYPES = frozenset(
    {"graph", "chart", "diagram", "table", "photo", "screenshot", "logo", "icon", "other"}
)


@dataclass(frozen=True)
class MistralOCRConfig:
    """Configuration for MistralOCR document processor."""

    # API key - falls back to MISTRAL_API_KEY env var if None.
    api_key: str | None = None
    # Whether to use vision to describe images in documents.
    include_image_descriptions: bool = True
    # Image types that Mistral classifies each image into.
    image_types: frozenset[str] = DEFAULT_IMAGE_TYPES
    # Image types to exclude from the output (e.g., {"logo", "icon"}).
    exclude_image_types: frozenset[str] = frozenset()
    model: str = "mistral-ocr-latest"


# The default instruction given to the vision language model when describing an image. It asks for
# a retrieval-oriented annotation rather than free-form prose: text transcription first (so OCR
# misses are recovered), then a factual description of the visual content.
DEFAULT_VLM_PROMPT = (
    "Transcribe all visible text in this image verbatim, then describe any charts, tables, "
    "diagrams, or other visual content. Be concise, factual, and complete. "
    "Answer in the same language as the text in the image."
)


@dataclass(frozen=True)
class MultimodalConfig:
    """Configuration for the multimodal (OCR + VLM) document processor.

    The processor handles images directly and PDFs by rasterising their pages, then runs two
    independent extraction stages over each page: OCR recovers text rendered inside the image,
    and a vision language model describes the visual content that OCR cannot capture.

    Both stages are optional and both engines are injectable, so the processor can run fully
    offline (`ocr_engine="auto"`, `vlm=None`), fully in the cloud (`vlm="openai/gpt-4o-mini"`),
    or with test doubles (`ocr_engine=<OCREngine>`, `vlm=<VLMEngine>`).
    """

    # OCR stage: "auto" picks the first installed engine (RapidOCR, then Tesseract), "none"
    # disables OCR, and an OCREngine instance is used as-is.
    ocr_engine: "OCREngine | str | None" = None
    # Languages requested from the RapidOCR engine.
    ocr_languages: tuple[str, ...] = ("ch", "en")
    # Language packs requested from the Tesseract engine.
    tesseract_languages: tuple[str, ...] = ("eng",)
    include_ocr_text: bool = True
    # Vision language model stage: a LiteLLM model identifier, a VLMEngine instance, or None to
    # skip image description entirely.
    vlm: "VLMEngine | str | None" = None
    vlm_api_key: str | None = None
    vlm_api_base: str | None = None
    vlm_prompt: str = DEFAULT_VLM_PROMPT
    vlm_max_tokens: int = 512
    vlm_temperature: float = 0.0
    vlm_num_retries: int = 3
    describe_images: bool = True
    # PDF rasterisation: resolution in dots per inch, and an optional page cap for large reports.
    pdf_dpi: int = 200
    max_pdf_pages: int | None = None
    # Number of worker threads used to extract the pages of a document in parallel.
    extraction_max_workers: int = 4


# The default instruction given to the entity extractor. It asks for a retrieval-oriented graph
# rather than an exhaustive one: a small number of high-value entities and their explicit
# relations, each grounded in the source text.
DEFAULT_GRAPH_EXTRACTION_PROMPT = (
    "You are a knowledge graph extractor with perfect precision and recall.\n"
    "Read the source text and extract the entities it mentions and the relations between them.\n"
    "Only extract entities and relations that are explicitly stated in the text.\n"
    "Use each entity's canonical name as written in the text; never invent or translate names.\n"
    "Give every entity a one-sentence description grounded in the text.\n"
    "Give every relation a short description of how the source entity relates to the target, and\n"
    "a strength from 1 (mentioned in passing) to 10 (the text is about this relation).\n"
    "Prefer a small number of high-value entities over an exhaustive list.\n"
    "Answer in the same language as the source text."
)

# The default entity and relation types offered to the extractor when the config declares none.
DEFAULT_GRAPH_ENTITY_TYPES = ("person", "organization", "location", "concept", "event", "other")
DEFAULT_GRAPH_RELATION_TYPES = ("is-a", "part-of", "causes", "authored-by", "related-to")


@dataclass(frozen=True)
class GraphConfig:
    """Configuration for the graph-enhanced retrieval path.

    This follows the LightRAG approach rather than WeKnora's: entities and relations are stored in
    the existing DuckDB/SQLModel database (no Neo4j, no extra service, and no new dependency), and
    the resulting chunk ranking joins reciprocal rank fusion as a third weighted ranking next to
    vector search and BM25.

    The extractor is injectable so the graph can be built fully offline (a deterministic rule-based
    extractor, useful for tests and ablations), fully in the cloud (an LLM identifier), or with
    test doubles — mirroring the OCREngine / VLMEngine injection pattern of MultimodalConfig.

    This path is off by default. Every chunk costs one extra LLM call at index time, and the graph
    ranking only adds value for relational or corpus-level questions, so it should be enabled only
    after benchmarking it against the two-path baseline on a real question set.
    """

    # Whether the graph ranking participates in hybrid search. Off by default (see above).
    enabled: bool = False
    # The entity extractor: None builds an LLM extractor from the RAGLite LLM config, a string is
    # used as the LiteLLM model identifier of an LLM extractor, and an EntityExtractor instance is
    # used as-is.
    extractor: "EntityExtractor | str | None" = field(default=None, compare=False)
    extraction_prompt: str = DEFAULT_GRAPH_EXTRACTION_PROMPT
    extraction_temperature: float = 0.1
    extraction_max_tokens: int = 1024
    extraction_max_workers: int = 4
    # Few-shot constraints injected into the extraction prompt, WeKnora's `ExtractConfig`: the
    # entity and relation types the extractor may use, and worked examples. Empty means "let the
    # model choose", which is the default because an over-constrained schema silently drops the
    # entities a personal corpus is actually about.
    entity_types: tuple[str, ...] = ()
    relation_types: tuple[str, ...] = ()
    extraction_examples: str = ""
    # Entities whose title is longer than this many characters are dropped (usually a whole
    # sentence rather than a name).
    max_entity_title_length: int = 120
    # Edge weighting, WeKnora's `graph.go`: the combination of pointwise mutual information (how
    # much more often two entities co-occur than chance) and the strength the extractor assigned
    # to the relation, normalised and scaled into the 1-10 range.
    pmi_weight: float = 0.6
    strength_weight: float = 0.4
    weight_scale: float = 9.0
    # The chunk graph: two chunks that share an entity are linked, which is what lets retrieval
    # walk from a chunk that mentions the query entity to a chunk that only mentions a related
    # one. WeKnora's `buildChunkGraph` does the same projection.
    build_chunk_graph: bool = True
    # How many chunks an entity may contribute to the projection. A hub entity mentioned by every
    # chunk would otherwise make the projection quadratic in the corpus size.
    max_chunks_per_entity: int = 64
    chunk_neighbor_hops: int = 1
    chunk_neighbor_weight: float = 0.5
    # How many of a chunk's neighbours the query expansion may follow. The projection links every
    # pair of chunks that share an entity, so a hub entity gives a chunk thousands of neighbours —
    # on NFCorpus the heaviest chunk has 2,852 — and following all of them costs millions of rows
    # per query and drags in most of the corpus. Only the heaviest edges are worth following.
    chunk_neighbor_limit: int = 32
    # Retrieval: how many seed entities to use, how many hops to expand over the entity relations,
    # and how much weight a neighbouring entity keeps relative to a directly matched one.
    seed_entity_limit: int = 8
    neighbor_hops: int = 1
    neighbor_weight: float = 0.5
    # Whether to ask the extractor for the question's entities at query time. Off by default
    # because it costs one extractor call per query; the lexical matcher needs no model call. When
    # enabled, the extracted entities are matched by title and unioned with the lexical matches.
    query_extraction: bool = False
    # Whether to drop the chunks the vector and keyword paths already found, WeKnora's
    # `filterSeenChunk`. Off by default: RRF already rewards a chunk that several paths agree on,
    # and the graph's value here is precisely to add chunks the other paths missed.
    deduplicate_seen_chunks: bool = False


# Lazily load the default search method to avoid circular imports.
# TODO: Replace with search_and_rerank_chunk_spans after benchmarking.
def _vector_search(
    query: str,
    *,
    num_results: int = 8,
    metadata_filter: MetadataFilter | None = None,
    config: "RAGLiteConfig | None" = None,
) -> tuple[list[ChunkId], list[float]]:
    from weknow._search import vector_search

    return vector_search(
        query, num_results=num_results, metadata_filter=metadata_filter, config=config
    )


@dataclass(frozen=True)
class RAGLiteConfig:
    """RAGLite config."""

    # Database config.
    db_url: str | URL = f"duckdb:///{(cache_path / 'weknow.db').as_posix()}"
    # LLM config used for generation.
    llm: str = field(
        default_factory=lambda: (
            "llama-cpp-python/unsloth/Qwen3-8B-GGUF/*Q4_K_M.gguf@8192"
            if llama_supports_gpu_offload()
            else "llama-cpp-python/unsloth/Qwen3-4B-GGUF/*Q4_K_M.gguf@8192"
        )
    )
    llm_max_tries: int = 4
    # Embedder config used for indexing.
    embedder: str = field(
        default_factory=lambda: (  # Nomic-embed may be better if only English is used.
            "llama-cpp-python/lm-kit/bge-m3-gguf/*F16.gguf@512"
            if llama_supports_gpu_offload() or (os.cpu_count() or 1) >= 4  # noqa: PLR2004
            else "llama-cpp-python/lm-kit/bge-m3-gguf/*Q4_K_M.gguf@512"
        )
    )
    embedder_normalize: bool = True
    # Chunk config used to partition documents into chunks.
    chunk_max_size: int = 2048  # Max number of characters per chunk.
    # Document processing config. None = default processor.
    document_processor: MistralOCRConfig | MultimodalConfig | None = None
    # The document parser engine to prefer ("builtin", "pandoc", "markitdown", "multimodal",
    # "mistral"). None lets the registry pick: the builtin engine, unless a document processor is
    # configured. An engine that does not support the file type, or whose dependency is missing,
    # falls back to the builtin engine instead of failing.
    parser_engine: str | None = None
    # Vector search config.
    vector_search_distance_metric: Literal["cosine", "dot", "l2"] = "cosine"
    vector_search_multivector: bool = True
    vector_search_query_adapter: bool = True  # Only supported for "cosine" and "dot" metrics.
    # Reranking config.
    reranker: BaseRanker | dict[str, BaseRanker] | None = field(
        default_factory=lambda: {
            "en": FlashRankRanker("ms-marco-MiniLM-L-12-v2", verbose=0, cache_dir=cache_path),
            "other": FlashRankRanker("ms-marco-MultiBERT-L-12", verbose=0, cache_dir=cache_path),
        },
        compare=False,  # Exclude the reranker from comparison to avoid lru_cache misses.
    )
    # Search config: you can pick any search method that returns (list[ChunkId], list[float]),
    # list[Chunk], or list[ChunkSpan].
    search_method: SearchMethod = field(default=_vector_search, compare=False)
    self_query: bool = False
    agentic_iterations: int = 3
    # Hybrid search config: RRF weights for the vector and keyword rankings.
    hybrid_vector_weight: float = 0.75
    hybrid_keyword_weight: float = 0.25
    # The RRF weight of the graph ranking. Deliberately smaller than the vector and keyword
    # weights so that a noisy graph ranking cannot dominate the fusion.
    hybrid_graph_weight: float = 0.1
    # Graph-enhanced retrieval config (entity/relation extraction and graph ranking).
    graph: GraphConfig = field(default_factory=GraphConfig, compare=False)
    # Low-confidence secondary retrieval: when the top rerank score is below the confidence
    # threshold, retrieve again with an expanded candidate pool (`secondary_retrieval_multiplier`
    # times the original pool). The confidence signal is the rerank score; without a reranker the
    # secondary pass is skipped.
    secondary_retrieval: bool = True
    rerank_confidence_threshold: float = 0.3
    secondary_retrieval_multiplier: int = 2
    # Query intent routing: route each query to the keyword or hybrid search method based on
    # lightweight heuristics (keyword-heavy short queries go to BM25, everything else to hybrid).
    search_router: bool = False
    # Retrieval result cache: TTL in seconds and max number of cached entries.
    retrieval_cache_ttl: float = 300.0
    retrieval_cache_max_size: int = 256
    # Model capability declarations for models that LiteLLM does not know about (third-party
    # gateways, private deployments, brand-new models). LiteLLM resolves the context size, the
    # function calling support, and the embedding dimension from its bundled cost map; models
    # missing from that map silently lose those capabilities, which makes the agentic search tool
    # unusable and forces a fallback embedding-dimension probe. Declaring them here restores the
    # real capabilities. Leave as None to let LiteLLM decide (with the conservative fallback).
    llm_context_size: int | None = None
    embedder_dimension: int | None = None
