"""
Integration test: Training with experiment tracking.

Uses an in-memory tracker (no external service) to verify the full
training → tracking lifecycle: start_run, log_config, log_metrics,
log_artifact, end_run.
"""

import pytest

from molfun.training import HeadOnlyFinetune
from molfun.tracking.base import BaseTracker
from molfun.tracking.composite import CompositeTracker
from tests.integration.conftest import build_custom_model, make_loader


class InMemoryTracker(BaseTracker):
    """Tracker that stores everything in memory for test assertions."""

    def __init__(self):
        self.runs: list[dict] = []
        self.configs: list[dict] = []
        self.metrics_log: list[tuple[dict, int | None]] = []
        self.artifacts: list[str] = []
        self.texts: list[tuple[str, str]] = []
        self.ended = False

    def start_run(self, name=None, tags=None, config=None):
        self.runs.append({"name": name, "tags": tags, "config": config})

    def log_metrics(self, metrics, step=None):
        self.metrics_log.append((dict(metrics), step))

    def log_config(self, config):
        self.configs.append(dict(config))

    def log_artifact(self, path, name=None):
        self.artifacts.append(path)

    def log_text(self, text, tag="log"):
        self.texts.append((text, tag))

    def end_run(self, status="completed"):
        self.ended = True


class TestSingleTracker:

    def test_tracker_receives_config_and_metrics(self):
        tracker = InMemoryTracker()
        model = build_custom_model()
        strategy = HeadOnlyFinetune(lr=1e-3, amp=False, loss_fn="mse")

        strategy.fit(
            model, make_loader(), epochs=3, verbose=False, tracker=tracker,
        )

        assert len(tracker.configs) == 1
        assert "strategy" in tracker.configs[0]

        assert len(tracker.metrics_log) == 3
        for metrics, step in tracker.metrics_log:
            assert "train_loss" in metrics
            assert "lr" in metrics
            assert step is not None


class TestCompositeTracker:

    def test_fans_out_to_both_trackers(self):
        t1 = InMemoryTracker()
        t2 = InMemoryTracker()
        composite = CompositeTracker([t1, t2])

        model = build_custom_model()
        strategy = HeadOnlyFinetune(lr=1e-3, amp=False, loss_fn="mse")

        strategy.fit(
            model, make_loader(), epochs=2, verbose=False, tracker=composite,
        )

        for t in (t1, t2):
            assert len(t.configs) == 1
            assert len(t.metrics_log) == 2

    def test_one_tracker_failure_does_not_break_other(self):

        class FailingTracker(BaseTracker):
            def start_run(self, name=None, tags=None, config=None):
                raise RuntimeError("boom")
            def log_metrics(self, metrics, step=None):
                raise RuntimeError("boom")
            def log_config(self, config):
                raise RuntimeError("boom")
            def log_artifact(self, path, name=None):
                raise RuntimeError("boom")
            def log_text(self, text, tag="log"):
                raise RuntimeError("boom")
            def end_run(self, status="completed"):
                raise RuntimeError("boom")

        good = InMemoryTracker()
        composite = CompositeTracker([FailingTracker(), good])

        model = build_custom_model()
        strategy = HeadOnlyFinetune(lr=1e-3, amp=False, loss_fn="mse")

        strategy.fit(
            model, make_loader(), epochs=1, verbose=False, tracker=composite,
        )

        assert len(good.configs) == 1
        assert len(good.metrics_log) == 1


class TestTrackerContextManager:

    def test_end_run_on_exit(self):
        tracker = InMemoryTracker()

        with tracker:
            tracker.start_run("test")
            tracker.log_metrics({"loss": 0.5}, step=1)

        assert tracker.ended
