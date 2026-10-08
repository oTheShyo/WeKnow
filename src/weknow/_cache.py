"""Retrieval result cache with TTL and bounded size."""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from weknow._typing import MetadataFilter

DEFAULT_CACHE_TTL = 300.0  # Seconds a cache entry stays valid.
DEFAULT_CACHE_MAX_SIZE = 256  # Max number of cached entries.


@dataclass
class _CacheEntry:
    """A single cached retrieval result."""

    value: tuple[list[str], list[float]]
    expires_at: float


@dataclass
class RetrievalCache:
    """A bounded TTL cache for retrieval results keyed by search parameters."""

    max_size: int = DEFAULT_CACHE_MAX_SIZE
    ttl: float = DEFAULT_CACHE_TTL
    entries: dict[str, _CacheEntry] = field(default_factory=dict)

    def get(self, key: str) -> tuple[list[str], list[float]] | None:
        """Get a cached result, or None if absent or expired."""
        entry = self.entries.get(key)
        if entry is None:
            return None
        if time.monotonic() >= entry.expires_at:
            del self.entries[key]
            return None
        return entry.value

    def set(self, key: str, value: tuple[list[str], list[float]], *, ttl: float | None = None) -> None:
        """Cache a result. Evict the oldest entry when the cache is full."""
        if len(self.entries) >= self.max_size and key not in self.entries:
            oldest_key = min(self.entries, key=lambda k: self.entries[k].expires_at)
            del self.entries[oldest_key]
        self.entries[key] = _CacheEntry(
            value=value, expires_at=time.monotonic() + (self.ttl if ttl is None else ttl)
        )

    def clear(self) -> None:
        """Invalidate all cached entries."""
        self.entries.clear()

    @staticmethod
    def make_key(
        query: str,
        *,
        num_results: int,
        method_name: str = "hybrid_search",
        metadata_filter: MetadataFilter | None = None,
    ) -> str:
        """Build a stable cache key from the search parameters."""
        normalized_filter = {
            key: sorted(value if isinstance(value, list) else [value])
            for key, value in (metadata_filter or {}).items()
        }
        key_material = json.dumps(
            {"query": query, "num_results": num_results, "method": method_name, "filter": normalized_filter},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(key_material.encode()).hexdigest()


# Process-wide default cache, shared by all cached search variants.
_default_cache = RetrievalCache()


def get_retrieval_cache() -> RetrievalCache:
    """Get the process-wide default retrieval cache."""
    return _default_cache


def cached_search(
    search: Any,
    query: str,
    *,
    num_results: int = 8,
    metadata_filter: MetadataFilter | None = None,
    config: Any = None,
) -> tuple[list[str], list[float]]:
    """Run a search method through the default retrieval cache."""
    cache = get_retrieval_cache()
    ttl = config.retrieval_cache_ttl if config is not None else None
    key = cache.make_key(query, num_results=num_results, method_name=search.__name__, metadata_filter=metadata_filter)
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = search(query, num_results=num_results, metadata_filter=metadata_filter, config=config)
    cache.set(key, (list(result[0]), list(result[1])), ttl=ttl)
    return list(result[0]), list(result[1])


def cached_hybrid_search(
    query: str,
    *,
    num_results: int = 8,
    metadata_filter: MetadataFilter | None = None,
    config: Any = None,
) -> tuple[list[str], list[float]]:
    """Hybrid search wrapped in the default retrieval cache. Usable as a config search method."""
    from weknow._search import hybrid_search

    return cached_search(
        hybrid_search, query, num_results=num_results, metadata_filter=metadata_filter, config=config
    )
