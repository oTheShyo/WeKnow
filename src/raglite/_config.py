"""RAGLite config."""

import contextlib
import os
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from platformdirs import user_data_dir
from sqlalchemy.engine import URL

from raglite._lazy_llama import llama_supports_gpu_offload
from raglite._typing import ChunkId, MetadataFilter, SearchMethod

if TYPE_CHECKING:
    from raglite._multimodal import OCREngine, VLMEngine

# Suppress rerankers output on import until [1] is fixed.
# [1] https://github.com/AnswerDotAI/rerankers/issues/36
with contextlib.redirect_stdout(StringIO()):
    from rerankers.models.flashrank_ranker import FlashRankRanker
    from rerankers.models.ranker import BaseRanker


cache_path = Path(user_data_dir("raglite", ensure_exists=True))


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


# Lazily load the default search method to avoid circular imports.
# TODO: Replace with search_and_rerank_chunk_spans after benchmarking.
def _vector_search(
    query: str,
    *,
    num_results: int = 8,
    metadata_filter: MetadataFilter | None = None,
    config: "RAGLiteConfig | None" = None,
) -> tuple[list[ChunkId], list[float]]:
    from raglite._search import vector_search

    return vector_search(
        query, num_results=num_results, metadata_filter=metadata_filter, config=config
    )


@dataclass(frozen=True)
class RAGLiteConfig:
    """RAGLite config."""

    # Database config.
    db_url: str | URL = f"duckdb:///{(cache_path / 'raglite.db').as_posix()}"
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
