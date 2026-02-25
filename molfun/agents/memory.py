"""
Persistent experiment memory for long-running agents.

Survives crashes, provides LLM-friendly summaries that fit in
context windows, and tracks the best experiment so far.
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import json

from molfun.agents.experiment import Experiment


class ExperimentMemory:
    """
    Append-only experiment journal with persistence and summarization.

    The agent writes experiments here as they complete. The memory
    provides compressed summaries for the LLM context window and
    persists everything to disk so the agent can resume after a crash.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.experiments: list[Experiment] = []
        self.reasoning_log: list[str] = []
        self._persist_path = Path(persist_path) if persist_path else None

        if self._persist_path and self._persist_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def log_experiment(self, experiment: Experiment) -> None:
        self.experiments.append(experiment)
        self._persist()

    def log_reasoning(self, text: str) -> None:
        self.reasoning_log.append(text)
        self._persist()

    def update_experiment(self, experiment_id: str, **updates) -> None:
        for exp in self.experiments:
            if exp.id == experiment_id:
                for k, v in updates.items():
                    if hasattr(exp, k):
                        setattr(exp, k, v)
                self._persist()
                return

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self.experiments)

    @property
    def completed(self) -> list[Experiment]:
        return [e for e in self.experiments if e.status == "completed"]

    @property
    def failed(self) -> list[Experiment]:
        return [e for e in self.experiments if e.status == "failed"]

    def best(self, metric: str = "val_loss", minimize: bool = True) -> Optional[Experiment]:
        candidates = self.completed
        if not candidates:
            return None

        def key(exp: Experiment):
            if metric == "val_loss":
                v = exp.best_val_loss
            else:
                v = exp.metrics.get(metric)
            if v is None:
                return float("inf") if minimize else float("-inf")
            return v

        return min(candidates, key=key) if minimize else max(candidates, key=key)

    def get(self, experiment_id: str) -> Optional[Experiment]:
        for exp in self.experiments:
            if exp.id == experiment_id:
                return exp
        return None

    def last(self, n: int = 5) -> list[Experiment]:
        return self.experiments[-n:]

    # ------------------------------------------------------------------
    # Summaries for LLM context
    # ------------------------------------------------------------------

    def summary_for_context(self, max_experiments: int = 10) -> str:
        """
        Compressed summary suitable for the LLM's context window.

        Includes overall stats, top experiments, recent experiments,
        and key patterns observed.
        """
        lines = []

        n_total = len(self.experiments)
        n_ok = len(self.completed)
        n_fail = len(self.failed)
        lines.append(
            f"EXPERIMENT MEMORY: {n_total} total ({n_ok} completed, {n_fail} failed)"
        )

        best_exp = self.best()
        if best_exp:
            lines.append(f"\nBEST SO FAR: {best_exp.summary_line()}")
            lines.append(f"  Config: {json.dumps(best_exp.config.to_dict(), default=str)}")

        if n_ok > 1:
            sorted_exps = sorted(
                self.completed,
                key=lambda e: e.best_val_loss if e.best_val_loss is not None else float("inf"),
            )
            lines.append(f"\nTOP {min(5, len(sorted_exps))} EXPERIMENTS:")
            for exp in sorted_exps[:5]:
                lines.append(f"  {exp.summary_line()}")

        recent = self.last(max_experiments)
        if recent:
            lines.append(f"\nLAST {len(recent)} EXPERIMENTS:")
            for exp in recent:
                lines.append(f"  {exp.summary_line()}")

        if self.failed:
            lines.append(f"\nRECENT FAILURES:")
            for exp in self.failed[-3:]:
                lines.append(f"  [{exp.id}] {exp.config.short_description()}: {exp.error}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "experiments": [e.to_dict() for e in self.experiments],
            "reasoning_log": self.reasoning_log[-50:],
        }
        self._persist_path.write_text(json.dumps(data, indent=2, default=str))

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text())
            self.experiments = [Experiment.from_dict(d) for d in data.get("experiments", [])]
            self.reasoning_log = data.get("reasoning_log", [])
        except (json.JSONDecodeError, KeyError):
            pass

    def save(self, path: str) -> None:
        old_path = self._persist_path
        self._persist_path = Path(path)
        self._persist()
        self._persist_path = old_path

    @classmethod
    def load(cls, path: str) -> ExperimentMemory:
        return cls(persist_path=path)
