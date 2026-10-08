"""Query intent routing.

Route each query to the search method that fits its intent, based on lightweight heuristics that
run in microseconds (no LLM call). Keyword-heavy short queries (exact terms, identifiers, quoted
phrases) are best served by BM25 keyword search; everything else (natural-language, semantic,
multi-facet questions) goes to hybrid search. The router is disabled by default and enabled with
`config.search_router = True`, or by assigning `routed_search` as the config search method.
"""

import re
from typing import Any

from weknow._typing import BasicSearchMethod, ChunkId, MetadataFilter

# Common English and Chinese function words, used to measure how "natural language" a query is.
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when", "where", "which", "who", "why", "will", "with", "的", "了", "吗", "呢", "吧", "是", "在", "和", "与", "有", "对", "关于", "怎么", "怎样", "什么", "哪些", "如何", "为什么", "请", "帮"]
)

_QUOTED_PHRASE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]+)[\"'“”‘’]")  # noqa: RUF001
_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def _tokenize(query: str) -> list[str]:
    """Lowercase tokens: alnum words and individual CJK characters."""
    return _TOKEN.findall(query.lower())


def route_search_method(
    query: str, *, config: Any = None
) -> BasicSearchMethod:
    """Route a query to the keyword or hybrid search method based on lightweight heuristics.

    Heuristics, in order:
    1. A quoted phrase is an exact-match intent: keyword search.
    2. A short query (6 tokens or fewer) with few function words (stopword ratio 0.25 or less) is
       keyword-heavy: keyword search.
    3. Everything else: hybrid search.
    """
    from weknow._search import hybrid_search, keyword_search

    # Quoted phrases signal exact-match intent.
    if _QUOTED_PHRASE.search(query):
        return keyword_search
    tokens = _tokenize(query)
    if not tokens:
        return hybrid_search
    stopword_ratio = sum(token in STOPWORDS for token in tokens) / len(tokens)
    # Short, keyword-heavy queries (few function words) are best served by BM25.
    if len(tokens) <= 6 and stopword_ratio <= 0.25:  # noqa: PLR2004
        return keyword_search
    return hybrid_search


def routed_search(
    query: str,
    *,
    num_results: int = 8,
    metadata_filter: MetadataFilter | None = None,
    config: Any = None,
) -> tuple[list[ChunkId], list[float]]:
    """Route the query to the fitting search method and run it. Usable as a config search method."""
    search_method = route_search_method(query, config=config)
    return search_method(query, num_results=num_results, metadata_filter=metadata_filter, config=config)
