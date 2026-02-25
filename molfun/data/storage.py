"""
Unified storage abstraction via fsspec.

Transparently handles local paths, S3, MinIO, GCS, Azure Blob,
HTTP, and any other filesystem fsspec supports. Local paths work
without fsspec installed (graceful fallback).

Usage::

    from molfun.data.storage import open_path, list_files, exists, ensure_dir

    # Local (no extra deps)
    with open_path("data/index.csv") as f:
        ...

    # S3
    with open_path("s3://bucket/index.csv") as f:
        ...

    # MinIO (S3-compatible)
    with open_path(
        "s3://bucket/index.csv",
        storage_options={"endpoint_url": "http://localhost:9000",
                         "key": "minioadmin", "secret": "minioadmin"},
    ) as f:
        ...

    # GCS
    with open_path("gs://bucket/index.csv") as f:
        ...

    # List files with glob
    files = list_files("s3://bucket/pdbs/*.cif")
"""

from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
import os

_HAS_FSSPEC = None


def _check_fsspec():
    global _HAS_FSSPEC
    if _HAS_FSSPEC is None:
        try:
            import fsspec  # noqa: F401
            _HAS_FSSPEC = True
        except ImportError:
            _HAS_FSSPEC = False
    return _HAS_FSSPEC


def is_remote(path: str) -> bool:
    """Check if a path is a remote URI (s3://, gs://, az://, http://, etc.)."""
    return "://" in str(path) and not str(path).startswith("file://")


def _get_fs(path: str, storage_options: Optional[dict] = None):
    """Get an fsspec filesystem for the given path."""
    if not _check_fsspec():
        raise ImportError(
            f"fsspec is required for remote path '{path}'. "
            "Install with: pip install 'molfun[streaming]'"
        )
    import fsspec
    opts = storage_options or {}
    return fsspec.core.url_to_fs(path, **opts)


@contextmanager
def open_path(
    path: str,
    mode: str = "r",
    storage_options: Optional[dict] = None,
):
    """
    Open a file from any fsspec-supported filesystem.

    For local paths, uses plain ``open()`` (no fsspec needed).
    For remote paths (s3://, gs://, etc.), uses fsspec.

    Args:
        path: Local or remote file path.
        mode: File mode ('r', 'rb', 'w', 'wb', etc.)
        storage_options: kwargs passed to the fsspec filesystem
            (e.g. ``{"endpoint_url": "http://localhost:9000"}`` for MinIO).

    Yields:
        File-like object.
    """
    if not is_remote(str(path)):
        local_path = Path(os.path.expanduser(str(path)))
        if "w" in mode:
            local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, mode) as f:
            yield f
    else:
        fs, fs_path = _get_fs(str(path), storage_options)
        with fs.open(fs_path, mode) as f:
            yield f


def list_files(
    pattern: str,
    storage_options: Optional[dict] = None,
) -> list[str]:
    """
    List files matching a glob pattern on any filesystem.

    Args:
        pattern: Glob pattern (e.g. "s3://bucket/pdbs/*.cif" or "data/*.csv")
        storage_options: fsspec options for remote filesystems.

    Returns:
        Sorted list of matching paths.
    """
    if not is_remote(pattern):
        local = Path(os.path.expanduser(pattern))
        parent = local.parent
        glob_part = local.name
        if parent.exists():
            return sorted(str(p) for p in parent.glob(glob_part))
        return []

    fs, fs_path = _get_fs(pattern, storage_options)
    protocol = str(pattern).split("://")[0]
    matches = fs.glob(fs_path)
    return sorted(f"{protocol}://{m}" for m in matches)


def exists(
    path: str,
    storage_options: Optional[dict] = None,
) -> bool:
    """Check if a file or directory exists on any filesystem."""
    if not is_remote(str(path)):
        return Path(os.path.expanduser(str(path))).exists()

    fs, fs_path = _get_fs(str(path), storage_options)
    return fs.exists(fs_path)


def ensure_dir(
    path: str,
    storage_options: Optional[dict] = None,
) -> None:
    """Create a directory (and parents) on any filesystem."""
    if not is_remote(str(path)):
        Path(os.path.expanduser(str(path))).mkdir(parents=True, exist_ok=True)
        return

    fs, fs_path = _get_fs(str(path), storage_options)
    fs.makedirs(fs_path, exist_ok=True)


def download_to_local(
    remote_path: str,
    local_path: str,
    storage_options: Optional[dict] = None,
) -> Path:
    """
    Download a remote file to a local path.

    Returns the local Path. If ``remote_path`` is already local,
    just returns it as a Path (no copy).
    """
    if not is_remote(str(remote_path)):
        return Path(os.path.expanduser(str(remote_path)))

    local = Path(os.path.expanduser(local_path))
    local.parent.mkdir(parents=True, exist_ok=True)

    fs, fs_path = _get_fs(str(remote_path), storage_options)
    fs.get(fs_path, str(local))
    return local
