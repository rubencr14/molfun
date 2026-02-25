"""Tests for molfun.benchmarks.metrics."""

import math

import pytest
import torch

from molfun.benchmarks.metrics import (
    MAE,
    RMSE,
    PearsonR,
    SpearmanRho,
    R2,
    AUROC,
    AUPRC,
    CoordRMSD,
    GDT_TS,
    LDDT,
    TM_Score,
    DockingSuccess,
    MetricCollection,
    METRIC_REGISTRY,
    create_metrics,
)


class TestMAE:
    def test_perfect(self):
        m = MAE()
        t = torch.tensor([1.0, 2.0, 3.0])
        m.update(t, t)
        assert m.compute()["mae"] == pytest.approx(0.0)

    def test_known_value(self):
        m = MAE()
        m.update(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0]))
        assert m.compute()["mae"] == pytest.approx(1.5)

    def test_accumulation(self):
        m = MAE()
        m.update(torch.tensor([1.0]), torch.tensor([2.0]))
        m.update(torch.tensor([3.0]), torch.tensor([3.0]))
        assert m.compute()["mae"] == pytest.approx(0.5)

    def test_reset(self):
        m = MAE()
        m.update(torch.tensor([1.0]), torch.tensor([10.0]))
        m.reset()
        m.update(torch.tensor([1.0]), torch.tensor([1.0]))
        assert m.compute()["mae"] == pytest.approx(0.0)


class TestRMSE:
    def test_perfect(self):
        m = RMSE()
        t = torch.tensor([1.0, 2.0])
        m.update(t, t)
        assert m.compute()["rmse"] == pytest.approx(0.0)

    def test_known_value(self):
        m = RMSE()
        m.update(torch.tensor([1.0, 3.0]), torch.tensor([1.0, 1.0]))
        assert m.compute()["rmse"] == pytest.approx(math.sqrt(2.0))


class TestPearsonR:
    def test_perfect_correlation(self):
        m = PearsonR()
        m.update(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]))
        assert m.compute()["pearson"] == pytest.approx(1.0, abs=1e-5)

    def test_negative_correlation(self):
        m = PearsonR()
        m.update(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 2.0, 1.0]))
        assert m.compute()["pearson"] == pytest.approx(-1.0, abs=1e-5)


class TestSpearmanRho:
    def test_perfect_rank(self):
        m = SpearmanRho()
        m.update(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([10.0, 20.0, 30.0]))
        assert m.compute()["spearman"] == pytest.approx(1.0, abs=1e-5)

    def test_inverse_rank(self):
        m = SpearmanRho()
        m.update(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([30.0, 20.0, 10.0]))
        assert m.compute()["spearman"] == pytest.approx(-1.0, abs=1e-5)


class TestR2:
    def test_perfect_fit(self):
        m = R2()
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        m.update(t, t)
        assert m.compute()["r2"] == pytest.approx(1.0, abs=1e-5)

    def test_poor_fit(self):
        m = R2()
        m.update(torch.tensor([2.5, 2.5, 2.5, 2.5]), torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert m.compute()["r2"] == pytest.approx(0.0, abs=1e-5)


class TestAUROC:
    def test_perfect_classification(self):
        m = AUROC()
        m.update(torch.tensor([0.9, 0.8, 0.2, 0.1]), torch.tensor([1.0, 1.0, 0.0, 0.0]))
        assert m.compute()["auroc"] == pytest.approx(1.0, abs=0.01)

    def test_random_classification(self):
        m = AUROC()
        torch.manual_seed(42)
        m.update(torch.rand(1000), (torch.rand(1000) > 0.5).float())
        assert 0.3 < m.compute()["auroc"] < 0.7


class TestAUPRC:
    def test_perfect_classification(self):
        m = AUPRC()
        m.update(torch.tensor([0.9, 0.8, 0.2, 0.1]), torch.tensor([1.0, 1.0, 0.0, 0.0]))
        assert m.compute()["auprc"] > 0.9


class TestCoordRMSD:
    def test_perfect_overlap(self):
        m = CoordRMSD()
        coords = torch.randn(10, 3)
        m.update(coords, coords)
        assert m.compute()["coord_rmsd"] == pytest.approx(0.0, abs=1e-6)

    def test_batched(self):
        m = CoordRMSD()
        p = torch.zeros(2, 5, 3)
        t = torch.ones(2, 5, 3)
        m.update(p, t)
        assert m.compute()["coord_rmsd"] == pytest.approx(math.sqrt(3.0), abs=1e-5)

    def test_with_mask(self):
        m = CoordRMSD()
        p = torch.zeros(1, 4, 3)
        t = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        m.update(p, t, mask=mask)
        assert m.compute()["coord_rmsd"] == pytest.approx(1.0, abs=1e-5)


class TestGDT_TS:
    def test_identical_structures(self):
        m = GDT_TS()
        coords = torch.randn(20, 3)
        m.update(coords, coords)
        assert m.compute()["gdt_ts"] == pytest.approx(100.0, abs=0.1)

    def test_distant_structures(self):
        m = GDT_TS()
        p = torch.zeros(20, 3)
        t = torch.ones(20, 3) * 100.0
        m.update(p, t)
        assert m.compute()["gdt_ts"] == pytest.approx(0.0, abs=0.1)


class TestLDDT:
    def test_identical(self):
        m = LDDT()
        coords = torch.randn(30, 3) * 5
        m.update(coords, coords)
        assert m.compute()["lddt"] == pytest.approx(100.0, abs=0.1)


class TestTM_Score:
    def test_identical(self):
        m = TM_Score()
        coords = torch.randn(50, 3) * 10
        m.update(coords, coords)
        assert m.compute()["tm_score"] == pytest.approx(1.0, abs=0.01)

    def test_short_sequence_skipped(self):
        m = TM_Score()
        m.update(torch.randn(3, 3), torch.randn(3, 3))
        assert m.compute()["tm_score"] == 0.0


class TestDockingSuccess:
    def test_close_pose(self):
        m = DockingSuccess(threshold=2.0)
        ref = torch.zeros(10, 3)
        pred = torch.ones(10, 3) * 0.1
        m.update(pred, ref)
        assert m.compute()["docking_success"] == 1.0

    def test_far_pose(self):
        m = DockingSuccess(threshold=2.0)
        ref = torch.zeros(10, 3)
        pred = torch.ones(10, 3) * 100.0
        m.update(pred, ref)
        assert m.compute()["docking_success"] == 0.0


class TestMetricCollection:
    def test_compose(self):
        mc = MetricCollection([MAE(), RMSE(), PearsonR()])
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        mc.update(preds, targets)
        result = mc.compute()
        assert "mae" in result
        assert "rmse" in result
        assert "pearson" in result
        assert result["mae"] == pytest.approx(0.0)

    def test_reset(self):
        mc = MetricCollection([MAE()])
        mc.update(torch.tensor([1.0]), torch.tensor([10.0]))
        mc.reset()
        mc.update(torch.tensor([1.0]), torch.tensor([1.0]))
        assert mc.compute()["mae"] == pytest.approx(0.0)


class TestRegistry:
    def test_all_registered(self):
        expected = {"mae", "rmse", "pearson", "spearman", "r2", "auroc", "auprc",
                    "coord_rmsd", "gdt_ts", "lddt", "tm_score", "docking_success"}
        assert expected.issubset(set(METRIC_REGISTRY.keys()))

    def test_create_metrics(self):
        mc = create_metrics(["mae", "pearson"])
        assert len(mc) == 2

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            create_metrics(["nonexistent"])
