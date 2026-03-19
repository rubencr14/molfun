"""
Integration test: Multi-model evaluation → Leaderboard → export.

Builds two architecturally different models, trains each briefly,
evaluates with metrics, and aggregates into a Leaderboard with
full serialization round-trip.
"""

import pytest
import torch

from molfun.training import HeadOnlyFinetune
from molfun.benchmarks.metrics import MAE, RMSE, MetricCollection
from molfun.benchmarks.report import BenchmarkReport, TaskResult, Leaderboard
from tests.integration.conftest import (
    build_custom_model, make_loader, SyntheticAffinityDataset,
    D_SINGLE, SEQ_LEN,
)


def _evaluate_model(model, loader) -> dict[str, float]:
    """Simple evaluation: run forward on dict batches and compute MAE/RMSE."""
    metrics = MetricCollection([MAE(), RMSE()])
    model.adapter.eval()
    model.head.eval()

    with torch.no_grad():
        for batch_data in loader:
            feats, targets = batch_data
            result = model.forward(feats)
            preds = result["preds"]
            metrics.update(preds.detach(), targets.detach())

    return metrics.compute()


class TestLeaderboardWorkflow:

    def test_full_workflow(self, tmp_path):
        train = make_loader(16)
        val = make_loader(8, batch_size=4)

        reports: list[BenchmarkReport] = []
        for name, block in [("Pairformer", "pairformer"), ("Transformer", "simple_transformer")]:
            block_cfg = {"d_single": D_SINGLE, "d_pair": 16, "n_heads": 4, "attention_cls": "standard"}
            if block == "simple_transformer":
                block_cfg = {"d_single": D_SINGLE, "n_heads": 4, "attention_cls": "gated"}
            model = build_custom_model(block=block, block_config=block_cfg)
            strategy = HeadOnlyFinetune(lr=1e-2, amp=False, loss_fn="mse")
            strategy.fit(model, train, epochs=3, verbose=False)

            eval_metrics = _evaluate_model(model, val)

            report = BenchmarkReport(model_name=name, suite_name="synthetic")
            report.results["affinity"] = TaskResult(
                task_name="affinity",
                metrics=eval_metrics,
                n_samples=8,
            )
            reports.append(report)

        lb = Leaderboard()
        for r in reports:
            lb.add(r)

        assert len(lb) == 2

        ranking = lb.rank("affinity", "mae", ascending=True)
        assert len(ranking) == 2
        assert ranking[0][1] <= ranking[1][1]

    def test_leaderboard_json_roundtrip(self, tmp_path):
        lb = Leaderboard()
        for name, mae in [("A", 0.3), ("B", 0.5)]:
            r = BenchmarkReport(model_name=name, suite_name="suite")
            r.results["task"] = TaskResult("task", {"mae": mae}, n_samples=10)
            lb.add(r)

        path = str(tmp_path / "lb.json")
        lb.save(path)

        lb2 = Leaderboard.load(path)
        assert len(lb2) == 2
        assert lb2.rank("task", "mae")[0][0] == "A"

    def test_leaderboard_markdown_table(self):
        lb = Leaderboard()
        for name, mae, rmse in [("X", 0.2, 0.3), ("Y", 0.4, 0.6)]:
            r = BenchmarkReport(model_name=name, suite_name="s")
            r.results["t"] = TaskResult("t", {"mae": mae, "rmse": rmse}, n_samples=5)
            lb.add(r)

        md = lb.table("t")
        assert "X" in md
        assert "Y" in md
        assert "mae" in md
        assert "rmse" in md

    def test_report_save_load(self, tmp_path):
        r = BenchmarkReport(model_name="M", suite_name="S")
        r.results["t1"] = TaskResult("t1", {"mae": 0.42, "rmse": 0.55}, n_samples=100)
        r.total_duration_s = 12.3

        path = str(tmp_path / "report.json")
        r.save(path)

        r2 = BenchmarkReport.load(path)
        assert r2.model_name == "M"
        assert r2.metric("t1", "mae") == pytest.approx(0.42)
        assert r2.total_duration_s == pytest.approx(12.3)
