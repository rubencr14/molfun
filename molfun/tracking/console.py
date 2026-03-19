"""
Console tracker — prints metrics to stdout.

Default tracker, no dependencies. Useful for local debugging
and as a fallback when no cloud tracker is configured.
"""

from __future__ import annotations
from typing import Optional
import json
import time

from molfun.tracking.base import BaseTracker


class ConsoleTracker(BaseTracker):
    """
    Logs everything to stdout.

    Usage::

        tracker = ConsoleTracker(prefix="[molfun]")
        tracker.start_run("baseline-pairformer")
        tracker.log_metrics({"train_loss": 0.5}, step=1)
    """

    def __init__(self, prefix: str = "[molfun]", verbose: bool = True):
        self.prefix = prefix
        self.verbose = verbose
        self._run_name: Optional[str] = None
        self._start_time: Optional[float] = None

    def start_run(self, name=None, tags=None, config=None):
        self._run_name = name or "unnamed"
        self._start_time = time.time()
        if self.verbose:
            tag_str = f" tags={tags}" if tags else ""
            print(f"{self.prefix} Run started: {self._run_name}{tag_str}")
        if config:
            self.log_config(config)

    def log_metrics(self, metrics, step=None):
        if not self.verbose:
            return
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in metrics.items()]
        step_str = f" step={step}" if step is not None else ""
        print(f"{self.prefix}{step_str} {', '.join(parts)}")

    def log_config(self, config):
        if self.verbose:
            print(f"{self.prefix} Config: {json.dumps(config, indent=2, default=str)}")

    def log_artifact(self, path, name=None):
        if self.verbose:
            display = name or path
            print(f"{self.prefix} Artifact: {display}")

    def log_text(self, text, tag="log"):
        if self.verbose:
            print(f"{self.prefix} [{tag}] {text}")

    def end_run(self, status="completed"):
        elapsed = time.time() - self._start_time if self._start_time else 0
        if self.verbose:
            print(f"{self.prefix} Run ended: {self._run_name} ({status}, {elapsed:.1f}s)")
        self._run_name = None
        self._start_time = None
