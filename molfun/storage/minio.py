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

    def _client(self):
        from minio import Minio
        return Minio(
            f"{self._endpoint}:{self._port}",
            access_key=self._access_key,
            secret_key=self._secret_key,
            secure=self._secure,
        )

    def sync_ids_to_local(
        self,
        pdb_ids: list[str],
        remote_prefix: str,
        local_dir: str,
        fmt: str = "cif",
    ) -> tuple[list[str], list[str]]:
        """
        Download specific PDB files from MinIO to local, by ID.

        Only downloads IDs that exist remotely and are missing locally.

        Returns:
            (found_locally, not_in_minio): lists of PDB IDs.
            found_locally = IDs now available on disk (pre-existing + downloaded).
            not_in_minio = IDs that don't exist in MinIO (need RCSB).
        """
        from pathlib import Path
        local = Path(local_dir)
        local.mkdir(parents=True, exist_ok=True)

        client = self._client()
        prefix = remote_prefix.lstrip("/")

        found = []
        missing = []
        for pid in pdb_ids:
            pid = pid.strip().lower()
            local_path = local / f"{pid}.{fmt}"
            if local_path.exists():
                found.append(pid)
                continue

            object_name = f"{prefix}/{pid}.{fmt}"
            try:
                client.fget_object(self._bucket, object_name, str(local_path))
                found.append(pid)
            except Exception:
                missing.append(pid)

        return found, missing

    def sync_to_remote(
        self,
        local_dir: str,
        remote_prefix: str,
        *,
        glob: str = "*.cif",
    ) -> int:
        """
        Upload local files to a MinIO prefix.

        Only uploads files that don't already exist remotely.
        Returns number of files uploaded.
        """
        from pathlib import Path
        local = Path(local_dir)

        client = self._client()
        prefix = remote_prefix.lstrip("/")

        existing = {
            obj.object_name.split("/")[-1]
            for obj in client.list_objects(self._bucket, prefix=prefix + "/", recursive=True)
        }

        uploaded = 0
        for f in sorted(local.glob(glob)):
            if f.name in existing:
                continue
            object_name = f"{prefix}/{f.name}"
            client.fput_object(self._bucket, object_name, str(f))
            uploaded += 1

        return uploaded
