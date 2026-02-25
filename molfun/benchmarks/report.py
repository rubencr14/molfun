"""
Benchmark report and leaderboard.

``BenchmarkReport`` is the output of a single evaluation run — one model
evaluated on one suite.  ``Leaderboard`` aggregates multiple reports for
comparison.

All objects are fully JSON-serializable and can export to Markdown, LaTeX,
and pandas DataFrames for papers and dashboards.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TaskResult:
    """Metrics for a single benchmark task."""

    task_name: str
    metrics: dict[str, float]
    n_samples: int = 0
    duration_s: float = 0.0

    def __getitem__(self, key: str) -> float:
        return self.metrics[key]


@dataclass
class BenchmarkReport:
    """
    Complete output of ``ModelEvaluator.run()``.

    Contains per-task results, model metadata, and export utilities.
    """

    model_name: str
    suite_name: str
    results: dict[str, TaskResult] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    total_duration_s: float = 0.0

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    def metric(self, task: str, metric_name: str) -> Optional[float]:
        """Get a single metric value.  Returns ``None`` if not found."""
        tr = self.results.get(task)
        if tr is None:
            return None
        return tr.metrics.get(metric_name)

    def all_metrics(self) -> dict[str, dict[str, float]]:
        """Flat dict: ``{task_name: {metric: value}}``."""
        return {name: tr.metrics for name, tr in self.results.items()}

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "suite_name": self.suite_name,
            "timestamp": self.timestamp,
            "total_duration_s": self.total_duration_s,
            "metadata": self.metadata,
            "results": {
                name: {
                    "metrics": tr.metrics,
                    "n_samples": tr.n_samples,
                    "duration_s": tr.duration_s,
                }
                for name, tr in self.results.items()
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkReport:
        results = {}
        for name, rd in d.get("results", {}).items():
            results[name] = TaskResult(
                task_name=name,
                metrics=rd["metrics"],
                n_samples=rd.get("n_samples", 0),
                duration_s=rd.get("duration_s", 0.0),
            )
        return cls(
            model_name=d["model_name"],
            suite_name=d["suite_name"],
            results=results,
            metadata=d.get("metadata", {}),
            timestamp=d.get("timestamp", ""),
            total_duration_s=d.get("total_duration_s", 0.0),
        )

    @classmethod
    def from_json(cls, s: str) -> BenchmarkReport:
        return cls.from_dict(json.loads(s))

    def save(self, path: str) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str) -> BenchmarkReport:
        return cls.from_json(Path(path).read_text())

    # ------------------------------------------------------------------
    # Export formats
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render as a Markdown table suitable for README or paper draft."""
        if not self.results:
            return f"*No results for {self.model_name}*"

        all_metric_names = _collect_metric_names(self.results)
        header = "| Task | " + " | ".join(all_metric_names) + " |"
        sep = "|---" + "|---" * len(all_metric_names) + "|"
        rows = []
        for name, tr in self.results.items():
            cells = [f"{tr.metrics.get(m, float('nan')):.4f}" for m in all_metric_names]
            rows.append(f"| {name} | " + " | ".join(cells) + " |")

        lines = [
            f"### {self.model_name} — {self.suite_name}",
            "",
            header,
            sep,
            *rows,
            "",
            f"*Evaluated {self.timestamp} in {self.total_duration_s:.1f}s*",
        ]
        return "\n".join(lines)

    def to_latex(self) -> str:
        """Render as a LaTeX tabular for academic papers."""
        if not self.results:
            return "% No results"

        all_metric_names = _collect_metric_names(self.results)
        n_cols = 1 + len(all_metric_names)
        col_spec = "l" + "c" * len(all_metric_names)

        lines = [
            r"\begin{table}[h]",
            rf"\caption{{{self.model_name} on {self.suite_name}}}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            "Task & " + " & ".join(all_metric_names) + r" \\",
            r"\midrule",
        ]
        for name, tr in self.results.items():
            cells = [f"{tr.metrics.get(m, float('nan')):.4f}" for m in all_metric_names]
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    def to_dataframe(self):
        """Convert to pandas DataFrame (pandas must be installed)."""
        import pandas as pd

        rows = []
        for name, tr in self.results.items():
            row = {"task": name, **tr.metrics, "n_samples": tr.n_samples, "duration_s": tr.duration_s}
            rows.append(row)
        return pd.DataFrame(rows).set_index("task")


# ------------------------------------------------------------------
# Leaderboard
# ------------------------------------------------------------------

class Leaderboard:
    """
    Aggregates ``BenchmarkReport`` instances for cross-model comparison.

    Persists to JSON for long-running experiment campaigns.
    """

    def __init__(self) -> None:
        self._reports: list[BenchmarkReport] = []

    def add(self, report: BenchmarkReport) -> None:
        self._reports.append(report)

    @property
    def reports(self) -> list[BenchmarkReport]:
        return list(self._reports)

    def rank(self, task: str, metric: str, ascending: bool = True) -> list[tuple[str, float]]:
        """
        Rank models by a specific metric on a specific task.

        Returns sorted list of ``(model_name, value)`` pairs.
        ``ascending=True`` means lower is better (e.g. MAE, RMSE).
        """
        entries: list[tuple[str, float]] = []
        for r in self._reports:
            v = r.metric(task, metric)
            if v is not None:
                entries.append((r.model_name, v))
        entries.sort(key=lambda x: x[1], reverse=not ascending)
        return entries

    def table(self, task: str, metrics: Optional[list[str]] = None) -> str:
        """Markdown comparison table for a given task."""
        if not self._reports:
            return "*No reports*"

        if metrics is None:
            for r in self._reports:
                tr = r.results.get(task)
                if tr:
                    metrics = list(tr.metrics.keys())
                    break
        if not metrics:
            return f"*No metrics for task {task}*"

        header = "| Model | " + " | ".join(metrics) + " |"
        sep = "|---" + "|---" * len(metrics) + "|"
        rows = []
        for r in self._reports:
            tr = r.results.get(task)
            if tr is None:
                continue
            cells = [f"{tr.metrics.get(m, float('nan')):.4f}" for m in metrics]
            rows.append(f"| {r.model_name} | " + " | ".join(cells) + " |")

        return "\n".join([header, sep, *rows])

    def to_dataframe(self):
        """Pivot table: rows=models, columns=task/metric combinations."""
        import pandas as pd

        rows = []
        for r in self._reports:
            row: dict = {"model": r.model_name}
            for tname, tr in r.results.items():
                for mname, val in tr.metrics.items():
                    row[f"{tname}/{mname}"] = val
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = [r.to_dict() for r in self._reports]
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> Leaderboard:
        data = json.loads(Path(path).read_text())
        lb = cls()
        for d in data:
            lb.add(BenchmarkReport.from_dict(d))
        return lb

    def __len__(self) -> int:
        return len(self._reports)

    def __repr__(self) -> str:
        names = [r.model_name for r in self._reports]
        return f"Leaderboard({names})"


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------

def _collect_metric_names(results: dict[str, TaskResult]) -> list[str]:
    """Deduplicated ordered list of all metric names across tasks."""
    seen: dict[str, None] = {}
    for tr in results.values():
        for m in tr.metrics:
            seen[m] = None
    return list(seen)
