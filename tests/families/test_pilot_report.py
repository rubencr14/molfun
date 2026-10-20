"""Pilot report: paired base-vs-tuned comparison, bootstrapped over UniProts."""

import numpy as np
import pandas as pd
import pytest

from molfun.families.kinase.pilot import pilot_report

METRICS = ["tm_mean", "lddt_mean", "rmsd_mean", "state_top1_correct", "state_recovered"]


def _per_target(uniprots, tm, lddt=None, top1=None, recovered=None):
    n = len(uniprots)
    return pd.DataFrame(
        {
            "structure_id": [str(i) for i in range(n)],
            "uniprot": uniprots,
            "tm_mean": tm,
            "lddt_mean": lddt if lddt is not None else np.full(n, 0.8),
            "rmsd_mean": np.full(n, 2.0),
            "state_top1_correct": top1 if top1 is not None else np.ones(n, bool),
            "state_recovered": recovered if recovered is not None else np.ones(n, bool),
        }
    )


def test_pilot_report_keys():
    base = _per_target([f"P{i}" for i in range(5)], np.full(5, 0.7))

    report = pilot_report(base, base.copy())

    assert {"n_targets", "n_uniprots", "underpowered", "go"} <= set(report)
    for key in ("tm", "lddt", "rmsd", "state_top1", "state_coverage@5"):
        assert set(report[key]) == {"base", "tuned", "delta", "ci95"}


def test_pilot_report_paired_delta_and_ci():
    uniprots = [f"P{i}" for i in range(10)]
    rng = np.random.default_rng(0)
    base_tm = rng.uniform(0.6, 0.9, 10)

    report = pilot_report(_per_target(uniprots, base_tm), _per_target(uniprots, base_tm + 0.05))

    assert report["tm"]["delta"] == pytest.approx(0.05)
    low, high = report["tm"]["ci95"]
    assert low > 0 and high == pytest.approx(0.05)


def test_pilot_report_bootstraps_over_uniprots_not_targets():
    """One protein with 20 targets weighs as much as one with a single target."""
    uniprots = ["BIG"] * 20 + [f"P{i}" for i in range(9)]
    base_tm = np.full(29, 0.5)
    tuned_tm = base_tm.copy()
    tuned_tm[:20] += 1.0  # only the big protein improves

    report = pilot_report(_per_target(uniprots, base_tm), _per_target(uniprots, tuned_tm))

    assert report["n_uniprots"] == 10
    assert report["tm"]["delta"] == pytest.approx(0.1)  # (1.0 + 9 × 0) / 10, not 20/29


def test_pilot_report_flags_underpowered_and_rejects_unpaired():
    small = _per_target([f"P{i}" for i in range(25)], np.full(25, 0.7))

    assert pilot_report(small, small.copy())["underpowered"] is True
    with pytest.raises(ValueError, match="same targets"):
        pilot_report(small, small.iloc[1:])


def test_pilot_go_needs_no_accuracy_loss_and_better_states():
    uniprots = [f"P{i}" for i in range(30)]
    base = _per_target(uniprots, np.full(30, 0.8), top1=np.zeros(30, bool))
    better_states = _per_target(uniprots, np.full(30, 0.8), top1=np.ones(30, bool))
    worse_fold = _per_target(uniprots, np.full(30, 0.7), top1=np.ones(30, bool))

    assert pilot_report(base, better_states)["go"] is True
    assert pilot_report(base, worse_fold)["go"] is False
