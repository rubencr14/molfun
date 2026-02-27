"""
Base class for object storage backends.

An ObjectStorage is a lightweight config object — it holds credentials
and a root URI, and exposes them in the format that PDBFetcher,
StreamingStructureDataset, and molfun.data.storage already expect.
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class ObjectStorage(ABC):
    """
    Abstract base for remote storage backends.

    Subclasses only need to provide:
      - uri              → root URI (e.g. "s3://bucket")
      - storage_options  → dict for fsspec (endpoint, credentials, ...)
    """

    @property
    @abstractmethod
    def uri(self) -> str:
        """Root URI of this storage (e.g. ``s3://molfun-data``)."""

    @property
    @abstractmethod
    def storage_options(self) -> dict:
        """fsspec-compatible options (endpoint, key, secret, …)."""

    def prefix(self, *parts: str) -> str:
        """Build a sub-path under the root URI."""
        base = self.uri.rstrip("/")
        for p in parts:
            base = f"{base}/{p.strip('/')}"
        return base

    @classmethod
    @abstractmethod
    def from_env(cls) -> "ObjectStorage":
        """Construct from environment variables."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(uri={self.uri!r})"
