"""
Integration test: Build → Train → Evaluate → Report.

End-to-end flow using ModelBuilder, HeadOnlyFinetune, and the
benchmarking report pipeline — all with small CPU models.
"""

import pytest
import torch

from tests.integration.conftest import build_custom_model, make_loader, N_SAMPLES
from molfun.training import HeadOnlyFinetune, FullFinetune
from molfun.benchmarks.report import BenchmarkReport, TaskResult, Leaderboard
from molfun.benchmarks.metrics import MAE, RMSE, MetricCollection


class TestBuildTrainEval:
    """Full pipeline: build model → train → produce report."""

    def test_headonly_train_produces_report(self):
        model = build_custom_model()
        strategy = HeadOnlyFinetune(lr=1e-3, amp=False, loss_fn="mse")

        history = strategy.fit(
            model, make_loader(), val_loader=make_loader(4, 2),
            epochs=2, verbose=False,
        )

        assert len(history) == 2
        assert all("train_loss" in h for h in history)
        assert all("val_loss" in h for h in history)

        report = BenchmarkReport(
            model_name="test-pairformer",
            suite_name="synthetic",
        )
        report.results["affinity"] = TaskResult(
            task_name="affinity",
            metrics={"train_loss": history[-1]["train_loss"]},
            n_samples=N_SAMPLES,
        )

        md = report.to_markdown()
        assert "test-pairformer" in md
        assert "affinity" in md

        json_str = report.to_json()
        roundtripped = BenchmarkReport.from_json(json_str)
        assert roundtripped.model_name == "test-pairformer"
        assert "affinity" in roundtripped.results

    def test_full_finetune_runs(self):
        model = build_custom_model()
        strategy = FullFinetune(lr=1e-4, amp=False, loss_fn="mse")

        history = strategy.fit(model, make_loader(), epochs=1, verbose=False)
        assert len(history) == 1
        assert history[0]["train_loss"] > 0

    def test_metric_collection_on_predictions(self):
        metrics = MetricCollection([MAE(), RMSE()])

        preds = torch.randn(20, 1)
        targets = preds + torch.randn(20, 1) * 0.1
        metrics.update(preds, targets)

        result = metrics.compute()
        assert "mae" in result
        assert "rmse" in result
        assert result["mae"] < result["rmse"] or result["mae"] == pytest.approx(result["rmse"], abs=0.5)

    def test_leaderboard_from_two_models(self):
        report_a = BenchmarkReport(model_name="ModelA", suite_name="synth")
        report_a.results["task1"] = TaskResult("task1", {"mae": 0.5, "rmse": 0.7}, n_samples=100)

        report_b = BenchmarkReport(model_name="ModelB", suite_name="synth")
        report_b.results["task1"] = TaskResult("task1", {"mae": 0.3, "rmse": 0.5}, n_samples=100)

        lb = Leaderboard()
        lb.add(report_a)
        lb.add(report_b)

        ranking = lb.rank("task1", "mae", ascending=True)
        assert ranking[0][0] == "ModelB"
        assert ranking[1][0] == "ModelA"

        md_table = lb.table("task1")
        assert "ModelA" in md_table
        assert "ModelB" in md_table

    def test_report_latex_export(self):
        report = BenchmarkReport(model_name="LaTeXTest", suite_name="suite")
        report.results["binding"] = TaskResult("binding", {"mae": 0.42}, n_samples=50)

        latex = report.to_latex()
        assert r"\begin{table}" in latex
        assert "0.42" in latex
        assert "LaTeXTest" in latex
