"""Tests for molfun.benchmarks.evaluator."""

import torch
import pytest

from molfun.benchmarks.evaluator import ModelEvaluator
from molfun.benchmarks.suites import BenchmarkSuite, BenchmarkTask, TaskType


class _MockModel:
    """Minimal model that returns synthetic predictions."""

    model_type = "mock"

    def __init__(self, dim=16):
        self._dim = dim

    def forward(self, batch, mask=None):
        B = 1
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    B = v.shape[0]
                    break
        return {"preds": torch.randn(B, 1)}


class _MockModelWithAdapter(_MockModel):
    class _Adapter:
        def eval(self):
            pass

    adapter = _Adapter()
    head = None

    def summary(self):
        return {"mock": True}


class TestModelEvaluator:
    def test_run_with_missing_data(self):
        """Tasks with nonexistent data sources produce NaN metrics."""
        model = _MockModelWithAdapter()
        suite = BenchmarkSuite.custom(
            "test",
            [BenchmarkTask(
                name="fake",
                data_source="/nonexistent/path",
                metrics=("mae", "pearson"),
            )],
        )
        evaluator = ModelEvaluator(model, suite, device="cpu")
        report = evaluator.run()

        assert "fake" in report.results
        import math
        assert math.isnan(report.results["fake"].metrics["mae"])
        assert report.results["fake"].n_samples == 0

    def test_report_metadata(self):
        model = _MockModelWithAdapter()
        suite = BenchmarkSuite.custom("test", [])
        evaluator = ModelEvaluator(model, suite, device="cpu", batch_size=4)
        report = evaluator.run()

        assert report.model_name == "mock"
        assert report.suite_name == "test"
        assert report.metadata["batch_size"] == 4
        assert report.metadata["device"] == "cpu"

    def test_multiple_tasks(self):
        model = _MockModelWithAdapter()
        suite = BenchmarkSuite.custom(
            "multi",
            [
                BenchmarkTask(name="t1", data_source="/fake1", metrics=("mae",)),
                BenchmarkTask(name="t2", data_source="/fake2", metrics=("rmse",)),
            ],
        )
        evaluator = ModelEvaluator(model, suite, device="cpu")
        report = evaluator.run()
        assert len(report.results) == 2
        assert "t1" in report.results
        assert "t2" in report.results
