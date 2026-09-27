"""RAGLite."""

from raglite._cache import cached_hybrid_search, get_retrieval_cache
from raglite._config import MistralOCRConfig, RAGLiteConfig
from raglite._database import Document
from raglite._delete import delete_documents, delete_documents_by_metadata
from raglite._eval import answer_evals, evaluate, insert_evals
from raglite._eval_generation import bleu_n, evaluate_generation, rouge_l_f1
from raglite._extract import expand_document_metadata
from raglite._incremental import diff_documents, sync_documents
from raglite._insert import insert_documents
from raglite._iterative_search import (
    search_and_rerank_chunk_spans_with_secondary_retrieval,
    search_with_secondary_retrieval,
)
from raglite._mistral_ocr import MistralOCRError
from raglite._query_adapter import update_query_adapter
from raglite._rag import add_context, async_rag, rag, retrieve_context
from raglite._router import route_search_method, routed_search
from raglite._search import (
    hybrid_search,
    keyword_search,
    rerank_chunks,
    rerank_chunks_with_scores,
    retrieve_chunk_spans,
    retrieve_chunks,
    search_and_rerank_chunk_spans,
    search_and_rerank_chunks,
    vector_search,
)

__all__ = [
    # Config
    "RAGLiteConfig",
    "MistralOCRConfig",
    "MistralOCRError",
    # Insert
    "Document",
    "insert_documents",
    # Incremental indexing
    "sync_documents",
    "diff_documents",
    # Delete
    "delete_documents",
    "delete_documents_by_metadata",
    # Extract
    "expand_document_metadata",
    # Search
    "hybrid_search",
    "keyword_search",
    "vector_search",
    "retrieve_chunks",
    "retrieve_chunk_spans",
    "rerank_chunks",
    "rerank_chunks_with_scores",
    "search_and_rerank_chunks",
    "search_and_rerank_chunk_spans",
    # Secondary retrieval
    "search_with_secondary_retrieval",
    "search_and_rerank_chunk_spans_with_secondary_retrieval",
    # Intent routing
    "route_search_method",
    "routed_search",
    # Cache
    "cached_hybrid_search",
    "get_retrieval_cache",
    # RAG
    "retrieve_context",
    "add_context",
    "async_rag",
    "rag",
    # Query adapter
    "update_query_adapter",
    # Evaluate
    "answer_evals",
    "insert_evals",
    "evaluate",
    # Generation evaluation
    "rouge_l_f1",
    "bleu_n",
    "evaluate_generation",
]
