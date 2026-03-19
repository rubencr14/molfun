"""
molfun.storage — object storage backends for datasets and artifacts.

Thin config wrappers that produce (uri, storage_options) for the existing
fsspec-based I/O in molfun.data.storage. No new I/O logic — just a clean
way to read credentials from the environment.

Usage
-----
    from molfun.storage import MinioStorage

    storage = MinioStorage.from_env()
    # → storage.uri            "s3://molfun-data/"
    # → storage.storage_options {"endpoint_url": ..., "key": ..., "secret": ...}

    fetcher = PDBFetcher(
        cache_dir=storage.prefix("pdbs"),
        storage_options=storage.storage_options,
    )
"""

from molfun.storage.base import ObjectStorage
from molfun.storage.minio import MinioStorage

__all__ = ["ObjectStorage", "MinioStorage"]
