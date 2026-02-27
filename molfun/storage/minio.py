"""
MinIO (S3-compatible) object storage backend.

Reads configuration from environment variables:

    MINIO_ENDPOINT   host (default: localhost)
    MINIO_PORT       port (default: 9000)
    MINIO_ACCESS_KEY access key (default: minioadmin)
    MINIO_SECRET_KEY secret key (default: minioadmin)
    MINIO_BUCKET     bucket name (default: molfun-data)
    MINIO_SECURE     "true" for HTTPS (default: false)

Compatible with any .env loader (python-dotenv, direnv, etc.).
"""

from __future__ import annotations
import os

from molfun.storage.base import ObjectStorage


class MinioStorage(ObjectStorage):
    """
    S3-compatible storage pointing at a MinIO instance.

    Usage::

        storage = MinioStorage.from_env()
        fetcher = PDBFetcher(
            cache_dir=storage.prefix("pdbs"),
            storage_options=storage.storage_options,
        )
    """

    def __init__(
        self,
        endpoint: str = "localhost",
        port: int = 9000,
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        bucket: str = "molfun-data",
        secure: bool = False,
    ):
        self._endpoint = endpoint
        self._port = port
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket
        self._secure = secure

    @classmethod
    def from_env(cls) -> "MinioStorage":
        """
        Build from MINIO_* environment variables.

        Automatically loads a .env file in the current working directory
        (or any parent) if python-dotenv is installed.
        """
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)
        except ImportError:
            pass
        return cls(
            endpoint=os.environ.get("MINIO_ENDPOINT", "localhost"),
            port=int(os.environ.get("MINIO_PORT", "9000")),
            access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
            bucket=os.environ.get("MINIO_BUCKET", "molfun-data"),
            secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
        )

    @property
    def uri(self) -> str:
        return f"s3://{self._bucket}"

    @property
    def storage_options(self) -> dict:
        scheme = "https" if self._secure else "http"
        return {
            "endpoint_url": f"{scheme}://{self._endpoint}:{self._port}",
            "key": self._access_key,
            "secret": self._secret_key,
        }

    def ensure_bucket(self) -> bool:
        """
        Create the bucket if it does not exist.

        Returns True if the bucket was created, False if it already existed.
        Raises ConnectionError if MinIO is unreachable.
        """
        from minio import Minio
        from minio.error import S3Error

        client = Minio(
            f"{self._endpoint}:{self._port}",
            access_key=self._access_key,
            secret_key=self._secret_key,
            secure=self._secure,
        )
        try:
            if not client.bucket_exists(self._bucket):
                client.make_bucket(self._bucket)
                return True
            return False
        except S3Error as e:
            raise RuntimeError(f"MinIO error while ensuring bucket '{self._bucket}': {e}") from e
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to MinIO at {self._endpoint}:{self._port} — {e}"
            ) from e
