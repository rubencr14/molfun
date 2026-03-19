"""
Embedding and prediction cache with memory and disk backends.

Keyed by a stable hash of the input (sequence string, PDB path + mtime,
or arbitrary dict). Supports TTL-based expiration and LRU eviction.

Three backends:
  - **memory**: Fast in-process LRU cache (lost on restart).
  - **disk**: Persistent cache in a local directory (survives restarts).
  - **tiered**: Memory L1 + Disk L2 (best of both).
"""

from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union, Literal
import hashlib
import json
import os
import time

import numpy as np


def _stable_hash(key: Union[str, dict, tuple]) -> str:
    """Produce a stable hex digest from a cache key."""
    if isinstance(key, str):
        raw = key.encode("utf-8")
    elif isinstance(key, dict):
        raw = json.dumps(key, sort_keys=True, default=str).encode("utf-8")
    elif isinstance(key, (list, tuple)):
        raw = json.dumps(list(key), sort_keys=True, default=str).encode("utf-8")
    else:
        raw = str(key).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class _MemoryBackend:
    """In-memory LRU cache backed by OrderedDict."""

    def __init__(self, max_size: int = 1024):
        self._max_size = max_size
        self._store: OrderedDict[str, tuple[np.ndarray, float]] = OrderedDict()

    def get(self, key: str, ttl: Optional[float]) -> Optional[np.ndarray]:
        if key not in self._store:
            return None
        value, ts = self._store[key]
        if ttl is not None and (time.time() - ts) > ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def put(self, key: str, value: np.ndarray) -> None:
        self._store[key] = (value, time.time())
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> int:
        n = len(self._store)
        self._store.clear()
        return n

    def __len__(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        return {"backend": "memory", "entries": len(self), "max_size": self._max_size}


class _DiskBackend:
    """Disk-based cache using numpy .npy files."""

    def __init__(self, directory: str):
        self._dir = Path(directory).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.npy"

    def _meta_path(self, key: str) -> Path:
        return self._dir / f"{key}.meta"

    def get(self, key: str, ttl: Optional[float]) -> Optional[np.ndarray]:
        p = self._path(key)
        if not p.exists():
            return None
        if ttl is not None:
            mp = self._meta_path(key)
            if mp.exists():
                ts = float(mp.read_text().strip())
                if (time.time() - ts) > ttl:
                    p.unlink(missing_ok=True)
                    mp.unlink(missing_ok=True)
                    return None
        return np.load(str(p))

    def put(self, key: str, value: np.ndarray) -> None:
        np.save(str(self._path(key)), value)
        self._meta_path(key).write_text(str(time.time()))

    def delete(self, key: str) -> bool:
        p = self._path(key)
        if p.exists():
            p.unlink()
            self._meta_path(key).unlink(missing_ok=True)
            return True
        return False

    def clear(self) -> int:
        n = 0
        for f in self._dir.glob("*.npy"):
            f.unlink()
            meta = f.with_suffix(".meta")
            meta.unlink(missing_ok=True)
            n += 1
        return n

    def __len__(self) -> int:
        return len(list(self._dir.glob("*.npy")))

    def stats(self) -> dict:
        total_bytes = sum(f.stat().st_size for f in self._dir.glob("*.npy"))
        return {
            "backend": "disk",
            "directory": str(self._dir),
            "entries": len(self),
            "size_mb": round(total_bytes / (1024 * 1024), 2),
        }


class EmbeddingCache:
    """
    Local cache for embeddings and predictions.

    Eliminates redundant GPU computation by caching numpy arrays
    keyed by a stable hash of the input.

    Args:
        backend: Cache backend — "memory", "disk", or "tiered".
        directory: Directory for disk storage
                   (default ``~/.molfun/embed_cache``).
        max_memory: Max entries for in-memory LRU (default 1024).
        ttl: Time-to-live in seconds. None = no expiration.

    Usage::

        cache = EmbeddingCache(backend="tiered")

        key = "MKWVTFISLLLLFSSAYS"
        hit = cache.get(key)
        if hit is None:
            embedding = model.embed(key)
            cache.put(key, embedding)
        else:
            embedding = hit
    """

    def __init__(
        self,
        backend: Literal["memory", "disk", "tiered"] = "tiered",
        directory: str = "~/.molfun/embed_cache",
        max_memory: int = 1024,
        ttl: Optional[float] = None,
    ):
        self.backend_type = backend
        self.ttl = ttl

        self._mem: Optional[_MemoryBackend] = None
        self._disk: Optional[_DiskBackend] = None

        if backend in ("memory", "tiered"):
            self._mem = _MemoryBackend(max_size=max_memory)
        if backend in ("disk", "tiered"):
            self._disk = _DiskBackend(directory=directory)

    def get(self, key: Union[str, dict]) -> Optional[np.ndarray]:
        """
        Look up a cached embedding.

        Args:
            key: Sequence string, PDB path, or arbitrary dict.

        Returns:
            Cached numpy array or None on miss.
        """
        h = _stable_hash(key)

        if self._mem is not None:
            hit = self._mem.get(h, self.ttl)
            if hit is not None:
                return hit

        if self._disk is not None:
            hit = self._disk.get(h, self.ttl)
            if hit is not None:
                if self._mem is not None:
                    self._mem.put(h, hit)
                return hit

        return None

    def put(self, key: Union[str, dict], value: np.ndarray) -> None:
        """
        Store an embedding in cache.

        Args:
            key: Sequence string, PDB path, or arbitrary dict.
            value: Numpy array to cache.
        """
        h = _stable_hash(key)

        if self._mem is not None:
            self._mem.put(h, value)
        if self._disk is not None:
            self._disk.put(h, value)

    def delete(self, key: Union[str, dict]) -> bool:
        """Remove a key from all cache layers. Returns True if found."""
        h = _stable_hash(key)
        found = False
        if self._mem is not None:
            found |= self._mem.delete(h)
        if self._disk is not None:
            found |= self._disk.delete(h)
        return found

    def clear(self) -> int:
        """Clear all entries. Returns total entries removed."""
        n = 0
        if self._mem is not None:
            n += self._mem.clear()
        if self._disk is not None:
            n += self._disk.clear()
        return n

    def stats(self) -> dict:
        """Return cache statistics."""
        info: dict = {"backend": self.backend_type, "ttl": self.ttl}
        if self._mem is not None:
            info["memory"] = self._mem.stats()
        if self._disk is not None:
            info["disk"] = self._disk.stats()
        return info

    def cached(self, func):
        """
        Decorator to cache function results.

        The first argument is used as the cache key::

            @cache.cached
            def embed(sequence: str) -> np.ndarray:
                return model.encode(sequence)

            result = embed("MKWVT...")  # computed
            result = embed("MKWVT...")  # from cache
        """
        def wrapper(key, *args, **kwargs):
            hit = self.get(key)
            if hit is not None:
                return hit
            result = func(key, *args, **kwargs)
            self.put(key, result)
            return result
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    def __len__(self) -> int:
        if self._disk is not None:
            return len(self._disk)
        if self._mem is not None:
            return len(self._mem)
        return 0

    def __repr__(self) -> str:
        return f"EmbeddingCache(backend={self.backend_type!r}, entries={len(self)})"
