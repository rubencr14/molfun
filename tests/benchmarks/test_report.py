"""Tests for molfun.benchmarks.report."""

import json
import tempfile
from pathlib import Path

from molfun.benchmarks.report import BenchmarkReport, TaskResult, Leaderboard


def _make_report(model_name: str = "test_model") -> BenchmarkReport:
    return BenchmarkReport(
        model_name=model_name,
        suite_name="test_suite",
        results={
            "task_a": TaskResult(task_name="task_a", metrics={"mae": 0.5, "pearson": 0.8}, n_samples=100),
            "task_b": TaskResult(task_name="task_b", metrics={"mae": 0.3, "rmse": 0.4}, n_samples=50),
        },
        total_duration_s=12.5,
    )


class TestBenchmarkReport:
    def test_metric_access(self):
        r = _make_report()
        assert r.metric("task_a", "mae") == 0.5
        assert r.metric("task_a", "nonexistent") is None
        assert r.metric("nonexistent_task", "mae") is None

    def test_all_metrics(self):
        r = _make_report()
        m = r.all_metrics()
        assert "task_a" in m
        assert m["task_a"]["pearson"] == 0.8

    def test_json_roundtrip(self):
        r = _make_report()
        j = r.to_json()
        r2 = BenchmarkReport.from_json(j)
        assert r2.model_name == "test_model"
        assert r2.metric("task_a", "mae") == 0.5
        assert r2.results["task_b"].n_samples == 50

    def test_save_load(self, tmp_path):
        r = _make_report()
        path = str(tmp_path / "report.json")
        r.save(path)
        r2 = BenchmarkReport.load(path)
        assert r2.suite_name == "test_suite"
        assert r2.metric("task_b", "rmse") == 0.4

    def test_to_markdown(self):
        r = _make_report()
        md = r.to_markdown()
        assert "test_model" in md
        assert "task_a" in md
        assert "0.5000" in md

    def test_to_latex(self):
        r = _make_report()
        tex = r.to_latex()
        assert r"\\begin{table}" in tex or "\\begin{table}" in tex
        assert "task_a" in tex


class TestLeaderboard:
    def test_add_and_rank(self):
        lb = Leaderboard()
        lb.add(_make_report("model_a"))
        r2 = _make_report("model_b")
        r2.results["task_a"].metrics["mae"] = 0.2
        lb.add(r2)

        ranking = lb.rank("task_a", "mae", ascending=True)
        assert ranking[0][0] == "model_b"
        assert ranking[0][1] == 0.2

    def test_table(self):
        lb = Leaderboard()
        lb.add(_make_report("model_a"))
        lb.add(_make_report("model_b"))
        tbl = lb.table("task_a")
        assert "model_a" in tbl
        assert "model_b" in tbl

    def test_persistence(self, tmp_path):
        lb = Leaderboard()
        lb.add(_make_report("m1"))
        lb.add(_make_report("m2"))
        path = str(tmp_path / "lb.json")
        lb.save(path)

        lb2 = Leaderboard.load(path)
        assert len(lb2) == 2
        assert lb2.reports[0].model_name == "m1"

    def test_empty_table(self):
        lb = Leaderboard()
        assert "No reports" in lb.table("anything")
