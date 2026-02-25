"""
Abstract experiment tracker.

All trackers implement the same interface so training loops,
agents, and evaluation scripts can log to any backend (or
multiple backends at once via CompositeTracker).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class BaseTracker(ABC):
    """
    Interface for experiment tracking.

    Lifecycle:
        tracker.start_run("my-experiment", tags=["lora", "pairformer"])
        tracker.log_config({"lr": 1e-4, "block": "pairformer"})
        for epoch in range(epochs):
            tracker.log_metrics({"train_loss": 0.5, "val_loss": 0.3}, step=epoch)
        tracker.log_artifact("checkpoints/best.pt")
        tracker.log_text("Training complete. Best val: 0.28", tag="summary")
        tracker.end_run(status="completed")
    """

    @abstractmethod
    def start_run(
        self,
        name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        config: Optional[dict] = None,
    ) -> None:
        """Start a new tracked run/experiment."""
        ...

    @abstractmethod
    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log scalar metrics (loss, accuracy, etc.)."""
        ...

    @abstractmethod
    def log_config(self, config: dict) -> None:
        """Log hyperparameters / experiment configuration."""
        ...

    @abstractmethod
    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """Log a file artifact (checkpoint, plot, etc.)."""
        ...

    @abstractmethod
    def log_text(self, text: str, tag: str = "log") -> None:
        """Log free-form text (agent reasoning, summaries, etc.)."""
        ...

    @abstractmethod
    def end_run(self, status: str = "completed") -> None:
        """End the current run."""
        ...

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end_run(status="completed")
