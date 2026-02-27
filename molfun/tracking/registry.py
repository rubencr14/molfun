"""
Experiment Registry — lightweight versioned experiment store.

Each run gets a UUID and is saved as a self-contained folder:

    experiments/<run_id>/
        manifest.json    ← config, hyperparams, GPU info, timing
        metrics.json     ← [{epoch, train_loss, val_loss, lr}, ...]
        pdb_refs.json    ← PDB IDs / paths used (references, not files)
        checkpoint/      ← model weights (uploaded via model.save())

The folder is written locally first, then optionally synced to any
ObjectStorage backend (MinIO, S3, etc.) using molfun.data.storage.

Usage::

    # Local only
    registry = ExperimentRegistry()

    # With MinIO
    from molfun.storage import MinioStorage
    registry = ExperimentRegistry(storage=MinioStorage.from_env())

    # In training
    registry.start_run("kinase-lora-v1", config={"rank": 8, "lr": 1e-4})
    registry.log_pdb_refs(pdb_paths)
    model.fit(train_loader, val_loader, strategy=strategy, tracker=registry)
    model.save(registry.checkpoint_path)
    registry.end_run()

    print(registry.run_id)   # UUID of the run
"""

from __future__ import annotations

import json
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from molfun.tracking.base import BaseTracker


_LOCAL_EXPERIMENTS_DIR = Path(".experiments")


class ExperimentRegistry(BaseTracker):
    """
    Lightweight experiment registry with optional remote sync.

    Stores every run under a UUID folder with structured JSON files.
    Optionally pushes to MinIO / S3 / any ObjectStorage backend.
    """

    def __init__(
        self,
        storage=None,
        local_dir: Optional[str] = None,
        run_prefix: str = "experiments",
    ):
        """
        Args:
            storage: Optional ObjectStorage (MinioStorage, etc.).
                     If provided, the run folder is also synced there.
            local_dir: Local base directory for experiment folders.
                       Defaults to ``.experiments/`` in the working dir.
            run_prefix: Sub-path inside the storage root (default "experiments").
        """
        self._storage = storage
        self._run_prefix = run_prefix
        self._base_local = Path(local_dir) if local_dir else _LOCAL_EXPERIMENTS_DIR

        self._run_id: Optional[str] = None
        self._run_dir: Optional[Path] = None
        self._metrics: list[dict] = []
        self._manifest: dict = {}
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    @property
    def run_dir(self) -> Optional[Path]:
        return self._run_dir

    @property
    def checkpoint_path(self) -> Optional[str]:
        """Path where model.save() should write the checkpoint."""
        if self._run_dir is None:
            return None
        return str(self._run_dir / "checkpoint")

    @property
    def remote_uri(self) -> Optional[str]:
        """Remote URI for this run (if storage configured)."""
        if self._storage is None or self._run_id is None:
            return None
        return self._storage.prefix(self._run_prefix, self._run_id)

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def start_run(
        self,
        name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        config: Optional[dict] = None,
    ) -> None:
        self._run_id = str(uuid.uuid4())
        self._run_dir = self._base_local / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()
        self._metrics = []

        self._manifest = {
            "run_id": self._run_id,
            "name": name or "unnamed",
            "tags": tags or [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda or "N/A",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
        if config:
            self._manifest["config"] = config

        self._write_json("manifest.json", self._manifest)

    def log_config(self, config: dict) -> None:
        if self._run_dir is None:
            self.start_run(config=config)
            return
        self._manifest.setdefault("config", {})
        self._manifest["config"].update(config)
        self._write_json("manifest.json", self._manifest)

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        if self._run_dir is None:
            self.start_run()
        entry = {"step": step, **{k: round(float(v), 6) for k, v in metrics.items()}}
        self._metrics.append(entry)
        self._write_json("metrics.json", self._metrics)

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """Record an artifact path in the manifest (does not copy files)."""
        if self._run_dir is None:
            self.start_run()
        self._manifest.setdefault("artifacts", [])
        self._manifest["artifacts"].append({"path": path, "name": name or Path(path).name})
        self._write_json("manifest.json", self._manifest)

    def log_text(self, text: str, tag: str = "log") -> None:
        if self._run_dir is None:
            self.start_run()
        (self._run_dir / f"{tag}.txt").write_text(text)

    def end_run(self, status: str = "completed") -> None:
        if self._run_dir is None:
            return
        elapsed = time.time() - (self._start_time or time.time())
        self._manifest["status"] = status
        self._manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
        self._manifest["elapsed_seconds"] = round(elapsed, 1)
        self._manifest["checkpoint_path"] = self.checkpoint_path
        if self._storage is not None:
            self._manifest["remote_uri"] = self.remote_uri
        self._write_json("manifest.json", self._manifest)
        self._write_json("metrics.json", self._metrics)

        if self._storage is not None:
            self._sync_to_remote()

    # ------------------------------------------------------------------
    # Extra: PDB references
    # ------------------------------------------------------------------

    def log_pdb_refs(self, paths: list[str]) -> None:
        """
        Log references to PDB structures used in training.

        Stores PDB IDs and paths — not the files themselves.
        """
        if self._run_dir is None:
            self.start_run()
        refs = [
            {"pdb_id": Path(p).stem.split(".")[0].upper(), "path": str(p)}
            for p in paths
        ]
        self._write_json("pdb_refs.json", refs)

    # ------------------------------------------------------------------
    # Remote sync
    # ------------------------------------------------------------------

    def _sync_to_remote(self) -> None:
        """Upload the entire run folder to remote storage via the minio client."""
        from minio import Minio

        st = self._storage
        client = Minio(
            f"{st._endpoint}:{st._port}",
            access_key=st._access_key,
            secret_key=st._secret_key,
            secure=st._secure,
        )
        bucket = st._bucket
        prefix = f"{self._run_prefix}/{self._run_id}"
        uploaded = 0

        for local_file in sorted(self._run_dir.rglob("*")):
            if not local_file.is_file():
                continue
            rel = local_file.relative_to(self._run_dir)
            object_name = f"{prefix}/{rel}"
            try:
                client.fput_object(bucket, object_name, str(local_file))
                uploaded += 1
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Failed to upload %s → %s/%s: %s",
                    local_file.name, bucket, object_name, e,
                )

        print(f"Synced {uploaded} files → s3://{bucket}/{prefix}/")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_json(self, filename: str, data) -> None:
        path = self._run_dir / filename
        path.write_text(json.dumps(data, indent=2, default=str))

    def __repr__(self) -> str:
        storage_info = self._storage.uri if self._storage else "local"
        return f"ExperimentRegistry(run_id={self._run_id!r}, storage={storage_info!r})"
