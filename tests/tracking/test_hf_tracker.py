"""Tests for HuggingFaceTracker and model_card generation."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

from molfun.tracking.model_card import generate_model_card


# ======================================================================
# Model card generation
# ======================================================================

class TestModelCard:
    def test_basic_card(self):
        summary = {"name": "openfold", "device": "cuda"}
        card = generate_model_card(summary)
        assert "---" in card
        assert "library_name: molfun" in card
        assert "openfold" in card

    def test_card_with_metrics(self):
        summary = {"name": "openfold", "device": "cuda"}
        metrics = {"mae": 0.42, "pearson": 0.85}
        card = generate_model_card(summary, metrics=metrics)
        assert "mae" in card
        assert "0.42" in card
        assert "Evaluation Results" in card

    def test_card_with_strategy(self):
        summary = {
            "name": "openfold",
            "device": "cuda",
            "strategy": {"name": "lora", "rank": 8},
        }
        card = generate_model_card(summary)
        assert "lora" in card

    def test_card_with_adapter_info(self):
        summary = {
            "name": "openfold",
            "device": "cuda",
            "adapter": {"total": 1000000, "trainable": 50000},
            "head": {"type": "AffinityHead", "params": 5000},
        }
        card = generate_model_card(summary)
        assert "1,000,000" in card
        assert "AffinityHead" in card

    def test_card_with_dataset(self):
        summary = {"name": "openfold", "device": "cuda"}
        card = generate_model_card(summary, dataset_name="PDBbind-v2020")
        assert "PDBbind-v2020" in card
        assert "datasets:" in card

    def test_card_with_tags(self):
        summary = {"name": "custom", "device": "cpu"}
        card = generate_model_card(summary, tags=["kinase", "affinity"])
        assert "kinase" in card

    def test_card_usage_section(self):
        summary = {"name": "openfold", "device": "cuda"}
        card = generate_model_card(summary)
        assert "from_hub" in card
        assert "```python" in card

    def test_card_has_yaml_frontmatter(self):
        summary = {"name": "openfold", "device": "cuda"}
        card = generate_model_card(summary)
        lines = card.split("\n")
        assert lines[0] == "---"
        closing_idx = [i for i, l in enumerate(lines[1:], 1) if l == "---"]
        assert len(closing_idx) >= 1


# ======================================================================
# HuggingFaceTracker (mocked HfApi)
# ======================================================================

class TestHuggingFaceTracker:
    @pytest.fixture
    def mock_hf_api(self):
        with patch("molfun.tracking.hf_tracker.HuggingFaceTracker.__init__", return_value=None) as _:
            pass

        mock_api = MagicMock()
        mock_api.create_repo = MagicMock()
        mock_api.upload_file = MagicMock()
        mock_api.upload_folder = MagicMock()
        mock_api.token = "fake-token"
        return mock_api

    def _make_tracker(self, mock_api):
        from molfun.tracking.hf_tracker import HuggingFaceTracker
        tracker = HuggingFaceTracker.__new__(HuggingFaceTracker)
        tracker.repo_id = "user/test-model"
        tracker.repo_type = "model"
        tracker._private = False
        tracker._api = mock_api
        tracker._run_name = None
        tracker._config = {}
        tracker._metrics = {}
        tracker._metrics_history = []
        tracker._tags = []
        tracker._texts = []
        tracker._repo_created = False
        return tracker

    def test_start_run(self, mock_hf_api):
        tracker = self._make_tracker(mock_hf_api)
        tracker.start_run(name="test-run", tags=["lora"], config={"lr": 1e-4})
        assert tracker._run_name == "test-run"
        assert tracker._tags == ["lora"]
        assert tracker._config["lr"] == 1e-4

    def test_log_metrics(self, mock_hf_api):
        tracker = self._make_tracker(mock_hf_api)
        tracker.log_metrics({"loss": 0.5}, step=1)
        tracker.log_metrics({"loss": 0.3}, step=2)
        assert tracker._metrics["loss"] == 0.3
        assert len(tracker._metrics_history) == 2

    def test_log_config(self, mock_hf_api):
        tracker = self._make_tracker(mock_hf_api)
        tracker.log_config({"model": "openfold"})
        tracker.log_config({"lr": 1e-4})
        assert tracker._config == {"model": "openfold", "lr": 1e-4}

    def test_log_artifact_file(self, mock_hf_api, tmp_path):
        tracker = self._make_tracker(mock_hf_api)
        f = tmp_path / "test.pt"
        f.write_text("data")
        tracker.log_artifact(str(f), name="model.pt")
        mock_hf_api.create_repo.assert_called_once()
        mock_hf_api.upload_file.assert_called_once()

    def test_log_artifact_dir(self, mock_hf_api, tmp_path):
        tracker = self._make_tracker(mock_hf_api)
        d = tmp_path / "checkpoint"
        d.mkdir()
        (d / "weights.pt").write_text("data")
        tracker.log_artifact(str(d), name="checkpoint")
        mock_hf_api.upload_folder.assert_called_once()

    def test_end_run_pushes_metadata(self, mock_hf_api):
        tracker = self._make_tracker(mock_hf_api)
        tracker._config = {"lr": 1e-4}
        tracker._metrics = {"loss": 0.3}
        tracker._metrics_history = [{"loss": 0.5, "step": 1}]
        tracker.end_run()
        assert mock_hf_api.upload_file.call_count == 3  # config + metrics + history

    def test_upload_model_card(self, mock_hf_api):
        tracker = self._make_tracker(mock_hf_api)
        tracker.upload_model_card("# Test Card")
        mock_hf_api.upload_file.assert_called_once()
        call_kwargs = mock_hf_api.upload_file.call_args
        assert call_kwargs.kwargs.get("path_in_repo") == "README.md" or \
               (len(call_kwargs.args) > 0 or "README.md" in str(call_kwargs))

    def test_log_text(self, mock_hf_api):
        tracker = self._make_tracker(mock_hf_api)
        tracker.log_text("some reasoning", tag="agent")
        assert tracker._texts == [("agent", "some reasoning")]

    def test_ensure_repo_called_once(self, mock_hf_api):
        tracker = self._make_tracker(mock_hf_api)
        tracker._ensure_repo()
        tracker._ensure_repo()
        mock_hf_api.create_repo.assert_called_once()


# ======================================================================
# CLI hub commands (help only, no real HF calls)
# ======================================================================

class TestHubCLI:
    def test_push_help(self):
        from typer.testing import CliRunner
        from molfun.cli import app
        result = CliRunner().invoke(app, ["push", "--help"])
        assert result.exit_code == 0
        assert "checkpoint" in result.output.lower()

    def test_pull_help(self):
        from typer.testing import CliRunner
        from molfun.cli import app
        result = CliRunner().invoke(app, ["pull", "--help"])
        assert result.exit_code == 0

    def test_push_dataset_help(self):
        from typer.testing import CliRunner
        from molfun.cli import app
        result = CliRunner().invoke(app, ["push-dataset", "--help"])
        assert result.exit_code == 0
        assert "dataset" in result.output.lower()
