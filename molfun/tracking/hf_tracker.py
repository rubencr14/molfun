"""
Hugging Face Hub tracker.

Implements BaseTracker to push artifacts (checkpoints, metrics,
model cards) to a Hugging Face Hub repository.

Unlike other trackers that focus on experiment dashboards, this
one produces a versioned, discoverable model repository with
model card, weights, and inference-ready metadata.

Requires: pip install huggingface_hub  (or pip install molfun[hub])
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import logging
import tempfile

from molfun.tracking.base import BaseTracker

logger = logging.getLogger(__name__)


class HuggingFaceTracker(BaseTracker):
    """
    Track experiments by pushing to Hugging Face Hub.

    Metrics and config are accumulated during the run and written
    to the repo on ``end_run()`` (or when ``log_artifact()`` is called).

    Usage::

        from molfun.tracking import HuggingFaceTracker

        tracker = HuggingFaceTracker(repo_id="user/my-model")
        tracker.start_run(name="lora-pairformer", config={...})
        tracker.log_metrics({"val_loss": 0.28}, step=10)
        tracker.log_artifact("runs/checkpoint/")
        tracker.end_run()

    Combine with other trackers::

        from molfun.tracking import CompositeTracker, WandbTracker

        tracker = CompositeTracker([
            WandbTracker(project="molfun"),
            HuggingFaceTracker(repo_id="user/my-model"),
        ])
    """

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        repo_type: str = "model",
    ):
        """
        Args:
            repo_id: HF Hub repo (e.g. "username/model-name").
            token: HF API token (or set HF_TOKEN env var).
            private: Whether the repo should be private.
            repo_type: "model" or "dataset".
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "huggingface_hub is required: pip install huggingface_hub "
                "or pip install molfun[hub]"
            )

        self.repo_id = repo_id
        self.repo_type = repo_type
        self._private = private
        self._api = HfApi(token=token)
        self._run_name: Optional[str] = None
        self._config: dict = {}
        self._metrics: dict = {}
        self._metrics_history: list[dict] = []
        self._tags: list[str] = []
        self._texts: list[tuple[str, str]] = []
        self._repo_created = False

    def _ensure_repo(self):
        if self._repo_created:
            return
        try:
            self._api.create_repo(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                private=self._private,
                exist_ok=True,
            )
            self._repo_created = True
        except Exception as e:
            logger.warning(f"Could not create/verify repo {self.repo_id}: {e}")
            self._repo_created = True

    def start_run(
        self,
        name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        config: Optional[dict] = None,
    ) -> None:
        self._run_name = name
        self._tags = tags or []
        if config:
            self._config.update(config)

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        self._metrics.update(metrics)
        entry = {**metrics}
        if step is not None:
            entry["step"] = step
        self._metrics_history.append(entry)

    def log_config(self, config: dict) -> None:
        self._config.update(config)

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """Upload a file or directory to the HF Hub repo."""
        self._ensure_repo()
        p = Path(path)

        if p.is_dir():
            self._api.upload_folder(
                repo_id=self.repo_id,
                folder_path=str(p),
                repo_type=self.repo_type,
                path_in_repo=name or p.name,
            )
            logger.info(f"Uploaded directory {p} → {self.repo_id}/{name or p.name}")
        elif p.is_file():
            self._api.upload_file(
                repo_id=self.repo_id,
                path_or_fileobj=str(p),
                path_in_repo=name or p.name,
                repo_type=self.repo_type,
            )
            logger.info(f"Uploaded {p} → {self.repo_id}/{name or p.name}")
        else:
            logger.warning(f"Artifact path not found: {path}")

    def log_text(self, text: str, tag: str = "log") -> None:
        self._texts.append((tag, text))

    def end_run(self, status: str = "completed") -> None:
        """Push accumulated metadata (config, metrics) to the repo."""
        self._ensure_repo()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            if self._config:
                config_path = tmp_path / "config.json"
                config_path.write_text(json.dumps(self._config, indent=2, default=str))
                self._api.upload_file(
                    repo_id=self.repo_id,
                    path_or_fileobj=str(config_path),
                    path_in_repo="config.json",
                    repo_type=self.repo_type,
                )

            if self._metrics:
                metrics_path = tmp_path / "metrics.json"
                metrics_path.write_text(json.dumps(self._metrics, indent=2, default=str))
                self._api.upload_file(
                    repo_id=self.repo_id,
                    path_or_fileobj=str(metrics_path),
                    path_in_repo="metrics.json",
                    repo_type=self.repo_type,
                )

            if self._metrics_history:
                history_path = tmp_path / "metrics_history.jsonl"
                with open(history_path, "w") as f:
                    for entry in self._metrics_history:
                        f.write(json.dumps(entry, default=str) + "\n")
                self._api.upload_file(
                    repo_id=self.repo_id,
                    path_or_fileobj=str(history_path),
                    path_in_repo="metrics_history.jsonl",
                    repo_type=self.repo_type,
                )

        logger.info(f"Run '{self._run_name}' metadata pushed to {self.repo_id}")

    def upload_model_card(self, card_text: str) -> None:
        """Push a README.md (model card) to the repo."""
        self._ensure_repo()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(card_text)
            f.flush()
            self._api.upload_file(
                repo_id=self.repo_id,
                path_or_fileobj=f.name,
                path_in_repo="README.md",
                repo_type=self.repo_type,
            )
        logger.info(f"Model card pushed to {self.repo_id}")

    def download_repo(self, local_dir: str, revision: Optional[str] = None) -> str:
        """Download the full repo to a local directory."""
        from huggingface_hub import snapshot_download
        return snapshot_download(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            local_dir=local_dir,
            revision=revision,
            token=self._api.token,
        )
