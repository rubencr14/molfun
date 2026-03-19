"""Tests for experiment memory and persistence."""

import json
import tempfile
from pathlib import Path
import pytest

from molfun.agents.experiment import ExperimentConfig, Experiment
from molfun.agents.memory import ExperimentMemory


def _make_experiment(exp_id: str, val_loss: float, status: str = "completed") -> Experiment:
    return Experiment(
        id=exp_id,
        config=ExperimentConfig(name=f"exp-{exp_id}"),
        status=status,
        history=[{"val_loss": val_loss}],
        duration_s=10.0,
    )


class TestExperimentMemory:
    def test_log_and_query(self):
        mem = ExperimentMemory()
        mem.log_experiment(_make_experiment("a", 0.5))
        mem.log_experiment(_make_experiment("b", 0.3))
        mem.log_experiment(_make_experiment("c", 0.7, "failed"))
        assert mem.count == 3
        assert len(mem.completed) == 2
        assert len(mem.failed) == 1

    def test_best(self):
        mem = ExperimentMemory()
        mem.log_experiment(_make_experiment("a", 0.5))
        mem.log_experiment(_make_experiment("b", 0.2))
        mem.log_experiment(_make_experiment("c", 0.8))
        best = mem.best()
        assert best.id == "b"

    def test_best_empty(self):
        mem = ExperimentMemory()
        assert mem.best() is None

    def test_get_by_id(self):
        mem = ExperimentMemory()
        mem.log_experiment(_make_experiment("x", 0.5))
        exp = mem.get("x")
        assert exp is not None
        assert exp.id == "x"
        assert mem.get("nonexistent") is None

    def test_last_n(self):
        mem = ExperimentMemory()
        for i in range(10):
            mem.log_experiment(_make_experiment(str(i), 0.5 - i * 0.01))
        recent = mem.last(3)
        assert len(recent) == 3
        assert recent[0].id == "7"

    def test_summary_for_context(self):
        mem = ExperimentMemory()
        mem.log_experiment(_make_experiment("a", 0.5))
        mem.log_experiment(_make_experiment("b", 0.3))
        summary = mem.summary_for_context()
        assert "EXPERIMENT MEMORY" in summary
        assert "BEST SO FAR" in summary

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "memory.json")

            mem1 = ExperimentMemory(persist_path=path)
            mem1.log_experiment(_make_experiment("a", 0.5))
            mem1.log_experiment(_make_experiment("b", 0.3))
            mem1.log_reasoning("Trying lower LR next")

            # Reload from disk
            mem2 = ExperimentMemory(persist_path=path)
            assert mem2.count == 2
            assert mem2.best().id == "b"
            assert "lower LR" in mem2.reasoning_log[0]

    def test_update_experiment(self):
        mem = ExperimentMemory()
        mem.log_experiment(_make_experiment("a", 0.5))
        mem.update_experiment("a", status="failed", error="OOM")
        exp = mem.get("a")
        assert exp.status == "failed"
        assert exp.error == "OOM"
