"""
Langfuse tracker — observability for LLM-powered agents + training.

Langfuse is particularly useful for Molfun agents since it traces
LLM calls alongside experiment metrics, giving full observability
of what the agent decided and why.

Requires: pip install langfuse

Usage::

    tracker = LangfuseTracker(
        public_key="pk-...",
        secret_key="sk-...",
        host="https://cloud.langfuse.com",  # or self-hosted
    )
    tracker.start_run("agent-search-001", tags=["lora", "pairformer"])
    tracker.log_metrics({"val_loss": 0.3}, step=1)
    tracker.log_text("Pairformer + flash attention best so far", tag="reasoning")
    tracker.end_run()
"""

from __future__ import annotations
from typing import Optional
import os

from molfun.tracking.base import BaseTracker


class LangfuseTracker(BaseTracker):
    """
    Log experiments and agent traces to Langfuse.

    Works with Langfuse Cloud or self-hosted instances.
    API keys can be passed directly or via environment variables
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST.
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ):
        try:
            from langfuse import Langfuse
        except ImportError:
            raise ImportError(
                "langfuse package required: pip install langfuse\n"
                "Or install with: pip install 'molfun[langfuse]'"
            )
        self._langfuse = Langfuse(
            public_key=public_key or os.environ.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=secret_key or os.environ.get("LANGFUSE_SECRET_KEY"),
            host=host or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        self._trace = None
        self._run_name: Optional[str] = None

    def start_run(self, name=None, tags=None, config=None):
        self._run_name = name or "molfun-run"
        self._trace = self._langfuse.trace(
            name=self._run_name,
            tags=tags or [],
            metadata=config or {},
        )

    def log_metrics(self, metrics, step=None):
        if self._trace is None:
            self.start_run()
        self._trace.span(
            name=f"metrics-step-{step}" if step is not None else "metrics",
            metadata=metrics,
        )

    def log_config(self, config):
        if self._trace is None:
            self.start_run(config=config)
        else:
            self._trace.update(metadata=config)

    def log_artifact(self, path, name=None):
        if self._trace is None:
            self.start_run()
        self._trace.span(
            name="artifact",
            metadata={"path": path, "name": name or path},
        )

    def log_text(self, text, tag="log"):
        if self._trace is None:
            self.start_run()
        self._trace.span(
            name=tag,
            input=text,
        )

    def end_run(self, status="completed"):
        if self._trace is not None:
            self._trace.update(
                metadata={"status": status},
            )
            self._langfuse.flush()
            self._trace = None
