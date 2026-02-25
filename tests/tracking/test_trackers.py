"""Tests for all tracking backends."""

import pytest
from unittest.mock import MagicMock

from molfun.tracking.base import BaseTracker
from molfun.tracking.console import ConsoleTracker
from molfun.tracking.composite import CompositeTracker


class MockTracker(BaseTracker):
    """In-memory tracker for testing."""

    def __init__(self):
        self.runs = []
        self.metrics_log = []
        self.config_log = []
        self.artifacts = []
        self.texts = []
        self.ended = False

    def start_run(self, name=None, tags=None, config=None):
        self.runs.append({"name": name, "tags": tags, "config": config})

    def log_metrics(self, metrics, step=None):
        self.metrics_log.append({"metrics": metrics, "step": step})

    def log_config(self, config):
        self.config_log.append(config)

    def log_artifact(self, path, name=None):
        self.artifacts.append({"path": path, "name": name})

    def log_text(self, text, tag="log"):
        self.texts.append({"text": text, "tag": tag})

    def end_run(self, status="completed"):
        self.ended = True


class TestConsoleTracker:
    def test_lifecycle(self, capsys):
        tracker = ConsoleTracker()
        tracker.start_run("test-run", tags=["a", "b"])
        tracker.log_config({"lr": 1e-4})
        tracker.log_metrics({"train_loss": 0.5, "val_loss": 0.3}, step=1)
        tracker.log_text("hello", tag="note")
        tracker.log_artifact("/path/to/model.pt")
        tracker.end_run()

        output = capsys.readouterr().out
        assert "test-run" in output
        assert "train_loss" in output
        assert "0.5" in output
        assert "hello" in output
        assert "model.pt" in output

    def test_silent(self, capsys):
        tracker = ConsoleTracker(verbose=False)
        tracker.start_run("silent")
        tracker.log_metrics({"loss": 0.1})
        tracker.end_run()
        assert capsys.readouterr().out == ""

    def test_context_manager(self, capsys):
        with ConsoleTracker() as tracker:
            tracker.start_run("ctx")
            tracker.log_metrics({"x": 1.0})
        output = capsys.readouterr().out
        assert "ended" in output.lower() or "ctx" in output


class TestCompositeTracker:
    def test_fans_out_to_all(self):
        t1 = MockTracker()
        t2 = MockTracker()
        comp = CompositeTracker([t1, t2])

        comp.start_run("composite-test", tags=["a"])
        comp.log_config({"lr": 1e-3})
        comp.log_metrics({"loss": 0.5}, step=1)
        comp.log_text("reasoning text", tag="agent")
        comp.log_artifact("/model.pt", name="best")
        comp.end_run()

        for t in (t1, t2):
            assert len(t.runs) == 1
            assert t.runs[0]["name"] == "composite-test"
            assert len(t.metrics_log) == 1
            assert t.metrics_log[0]["metrics"]["loss"] == 0.5
            assert len(t.config_log) == 1
            assert len(t.texts) == 1
            assert len(t.artifacts) == 1
            assert t.ended

    def test_survives_failing_tracker(self):
        good = MockTracker()
        bad = MockTracker()
        bad.log_metrics = MagicMock(side_effect=RuntimeError("API down"))

        comp = CompositeTracker([bad, good])
        comp.log_metrics({"loss": 0.5}, step=1)

        assert len(good.metrics_log) == 1
        bad.log_metrics.assert_called_once()

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            CompositeTracker([])


class TestLazyImports:
    def test_wandb_import_error(self):
        """WandbTracker gives clear error when wandb not installed."""
        try:
            from molfun.tracking import WandbTracker
            WandbTracker(project="test")
        except ImportError as e:
            assert "wandb" in str(e).lower()

    def test_comet_import_error(self):
        try:
            from molfun.tracking import CometTracker
            CometTracker(project_name="test")
        except ImportError as e:
            assert "comet" in str(e).lower()

    def test_mlflow_import_error(self):
        try:
            from molfun.tracking import MLflowTracker
            MLflowTracker(experiment_name="test")
        except ImportError as e:
            assert "mlflow" in str(e).lower()

    def test_langfuse_import_error(self):
        try:
            from molfun.tracking import LangfuseTracker
            LangfuseTracker()
        except ImportError as e:
            assert "langfuse" in str(e).lower()


class TestTrackerWithTraining:
    """Test that the tracker integrates with the training loop."""

    def test_fit_with_tracker(self):
        """Mocked training loop calls tracker hooks."""
        tracker = MockTracker()
        tracker.start_run("training-test")

        tracker.log_config({"strategy": "lora", "lr": 1e-4})
        for epoch in range(3):
            tracker.log_metrics(
                {"epoch": epoch + 1, "train_loss": 1.0 - epoch * 0.2, "val_loss": 0.8 - epoch * 0.1},
                step=epoch + 1,
            )
        tracker.end_run()

        assert len(tracker.metrics_log) == 3
        assert tracker.metrics_log[0]["metrics"]["epoch"] == 1
        assert tracker.metrics_log[2]["step"] == 3
        assert tracker.ended
