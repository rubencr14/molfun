"""
Weights & Biases (W&B) tracker.

Requires: pip install wandb
"""

from __future__ import annotations
from typing import Optional

from molfun.tracking.base import BaseTracker


class WandbTracker(BaseTracker):
    """
    Log experiments to Weights & Biases.

    Usage::

        tracker = WandbTracker(project="molfun-search", entity="my-team")
        tracker.start_run("baseline-pairformer", tags=["lora"])
        tracker.log_metrics({"val_loss": 0.3}, step=1)
        tracker.end_run()
    """

    def __init__(
        self,
        project: str = "molfun",
        entity: Optional[str] = None,
        **wandb_init_kwargs,
    ):
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb package required: pip install wandb\n"
                "Or install with: pip install 'molfun[wandb]'"
            )
        self.project = project
        self.entity = entity
        self._init_kwargs = wandb_init_kwargs
        self._run = None

    def start_run(self, name=None, tags=None, config=None):
        kwargs = {
            "project": self.project,
            "name": name,
            "tags": tags,
            "config": config or {},
            **self._init_kwargs,
        }
        if self.entity:
            kwargs["entity"] = self.entity
        self._run = self._wandb.init(**kwargs)

    def log_metrics(self, metrics, step=None):
        if self._run is None:
            self.start_run()
        log_kwargs = dict(metrics)
        if step is not None:
            log_kwargs["step"] = step
        self._wandb.log(log_kwargs)

    def log_config(self, config):
        if self._run is None:
            self.start_run(config=config)
        else:
            self._wandb.config.update(config, allow_val_change=True)

    def log_artifact(self, path, name=None):
        if self._run is None:
            self.start_run()
        artifact = self._wandb.Artifact(
            name=name or "model-checkpoint",
            type="model",
        )
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def log_text(self, text, tag="log"):
        if self._run is None:
            self.start_run()
        self._wandb.log({tag: self._wandb.Html(f"<pre>{text}</pre>")})

    def end_run(self, status="completed"):
        if self._run is not None:
            self._run.finish(exit_code=0 if status == "completed" else 1)
            self._run = None
