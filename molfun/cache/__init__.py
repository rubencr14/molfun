"""
Local caching for embeddings and predictions.

Avoids redundant GPU computation by caching results keyed
by sequence hash. Supports in-memory (LRU), disk, and
combined (memory + disk tiered) backends.

Usage::

    from molfun.cache import EmbeddingCache

    cache = EmbeddingCache(backend="disk", directory="~/.molfun/embed_cache")
    cache = EmbeddingCache(backend="memory", max_size=1024)
    cache = EmbeddingCache(backend="tiered")  # memory + disk
"""

from molfun.cache.embedding import EmbeddingCache

__all__ = ["EmbeddingCache"]
