"""
MLflow tracker.

Requires: pip install mlflow
"""

from __future__ import annotations
from typing import Optional

from molfun.tracking.base import BaseTracker


class MLflowTracker(BaseTracker):
    """
    Log experiments to MLflow.

    Usage::

        tracker = MLflowTracker(
            experiment_name="molfun-search",
            tracking_uri="http://localhost:5000",  # or local ./mlruns
        )
        tracker.start_run("baseline-pairformer", tags=["lora"])
        tracker.log_metrics({"val_loss": 0.3}, step=1)
        tracker.end_run()
    """

    def __init__(
        self,
        experiment_name: str = "molfun",
        tracking_uri: Optional[str] = None,
    ):
        try:
            import mlflow
            self._mlflow = mlflow
        except ImportError:
            raise ImportError(
                "mlflow package required: pip install mlflow\n"
                "Or install with: pip install 'molfun[mlflow]'"
            )
        if tracking_uri:
            self._mlflow.set_tracking_uri(tracking_uri)
        self._mlflow.set_experiment(experiment_name)
        self._run = None

    def start_run(self, name=None, tags=None, config=None):
        tag_dict = {}
        if tags:
            for t in tags:
                tag_dict[t] = "true"
        self._run = self._mlflow.start_run(run_name=name, tags=tag_dict or None)
        if config:
            self._mlflow.log_params(
                {k: str(v)[:250] for k, v in config.items()}
            )

    def log_metrics(self, metrics, step=None):
        if self._run is None:
            self.start_run()
        self._mlflow.log_metrics(metrics, step=step)

    def log_config(self, config):
        if self._run is None:
            self.start_run(config=config)
        else:
            self._mlflow.log_params(
                {k: str(v)[:250] for k, v in config.items()}
            )

    def log_artifact(self, path, name=None):
        if self._run is None:
            self.start_run()
        self._mlflow.log_artifact(path)

    def log_text(self, text, tag="log"):
        if self._run is None:
            self.start_run()
        self._mlflow.log_text(text, f"{tag}.txt")

    def end_run(self, status="completed"):
        if self._run is not None:
            mlflow_status = "FINISHED" if status == "completed" else "FAILED"
            self._mlflow.end_run(status=mlflow_status)
            self._run = None
