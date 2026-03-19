"""
Comet ML tracker.

Requires: pip install comet-ml
"""

from __future__ import annotations
from typing import Optional

from molfun.tracking.base import BaseTracker


class CometTracker(BaseTracker):
    """
    Log experiments to Comet ML.

    Usage::

        tracker = CometTracker(
            project_name="molfun-search",
            api_key="your-key",  # or set COMET_API_KEY env var
        )
        tracker.start_run("baseline-pairformer", tags=["lora"])
        tracker.log_metrics({"val_loss": 0.3}, step=1)
        tracker.end_run()
    """

    def __init__(
        self,
        project_name: str = "molfun",
        workspace: Optional[str] = None,
        api_key: Optional[str] = None,
        **comet_kwargs,
    ):
        try:
            import comet_ml
            self._comet_ml = comet_ml
        except ImportError:
            raise ImportError(
                "comet-ml package required: pip install comet-ml\n"
                "Or install with: pip install 'molfun[comet]'"
            )
        self.project_name = project_name
        self.workspace = workspace
        self.api_key = api_key
        self._comet_kwargs = comet_kwargs
        self._experiment = None

    def start_run(self, name=None, tags=None, config=None):
        kwargs = {
            "project_name": self.project_name,
            **self._comet_kwargs,
        }
        if self.workspace:
            kwargs["workspace"] = self.workspace
        if self.api_key:
            kwargs["api_key"] = self.api_key

        self._experiment = self._comet_ml.Experiment(**kwargs)
        if name:
            self._experiment.set_name(name)
        if tags:
            self._experiment.add_tags(tags)
        if config:
            self._experiment.log_parameters(config)

    def log_metrics(self, metrics, step=None):
        if self._experiment is None:
            self.start_run()
        for key, value in metrics.items():
            self._experiment.log_metric(key, value, step=step)

    def log_config(self, config):
        if self._experiment is None:
            self.start_run(config=config)
        else:
            self._experiment.log_parameters(config)

    def log_artifact(self, path, name=None):
        if self._experiment is None:
            self.start_run()
        self._experiment.log_asset(path, file_name=name)

    def log_text(self, text, tag="log"):
        if self._experiment is None:
            self.start_run()
        self._experiment.log_text(text, metadata={"tag": tag})

    def end_run(self, status="completed"):
        if self._experiment is not None:
            self._experiment.end()
            self._experiment = None
