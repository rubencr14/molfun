"""
Composite tracker — logs to multiple backends simultaneously.

Usage::

    tracker = CompositeTracker([
        ConsoleTracker(),
        WandbTracker(project="molfun"),
    ])
    tracker.start_run("experiment-1")
    tracker.log_metrics({"val_loss": 0.3}, step=1)
    # → logged to both console AND W&B
"""

from __future__ import annotations
from typing import Optional

from molfun.tracking.base import BaseTracker


class CompositeTracker(BaseTracker):
    """
    Fans out every call to multiple trackers.

    If one tracker fails, the others still receive the call.
    Errors are logged but not raised to avoid interrupting training.
    """

    def __init__(self, trackers: list[BaseTracker]):
        if not trackers:
            raise ValueError("CompositeTracker needs at least one tracker")
        self.trackers = list(trackers)

    def start_run(self, name=None, tags=None, config=None):
        for t in self.trackers:
            try:
                t.start_run(name=name, tags=tags, config=config)
            except Exception as e:
                _warn(t, "start_run", e)

    def log_metrics(self, metrics, step=None):
        for t in self.trackers:
            try:
                t.log_metrics(metrics, step=step)
            except Exception as e:
                _warn(t, "log_metrics", e)

    def log_config(self, config):
        for t in self.trackers:
            try:
                t.log_config(config)
            except Exception as e:
                _warn(t, "log_config", e)

    def log_artifact(self, path, name=None):
        for t in self.trackers:
            try:
                t.log_artifact(path, name=name)
            except Exception as e:
                _warn(t, "log_artifact", e)

    def log_text(self, text, tag="log"):
        for t in self.trackers:
            try:
                t.log_text(text, tag=tag)
            except Exception as e:
                _warn(t, "log_text", e)

    def end_run(self, status="completed"):
        for t in self.trackers:
            try:
                t.end_run(status=status)
            except Exception as e:
                _warn(t, "end_run", e)


def _warn(tracker: BaseTracker, method: str, error: Exception) -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        "Tracker %s.%s failed: %s",
        type(tracker).__name__, method, error,
    )
