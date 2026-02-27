"""Tests for pretrained model registry, from_pretrained, and predict(sequence)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch
import torch.nn as nn


class TestPretrainedRegistry:
    def test_registry_has_openfold(self):
        from molfun.hub.registry import PRETRAINED_REGISTRY
        assert "openfold" in PRETRAINED_REGISTRY
        assert "openfold_ptm" in PRETRAINED_REGISTRY

    def test_spec_fields(self):
        from molfun.hub.registry import PRETRAINED_REGISTRY
        spec = PRETRAINED_REGISTRY["openfold"]
        assert spec.backend == "openfold"
        assert spec.weights_url.startswith("https://")
        assert spec.weights_filename.endswith(".pt")
        assert spec.config_preset == "finetuning_ptm"

    def test_list_pretrained(self):
        from molfun.hub.registry import list_pretrained
        specs = list_pretrained()
        assert len(specs) >= 2
        names = [s.name for s in specs]
        assert "openfold" in names

    def test_download_unknown_model_raises(self):
        from molfun.hub.registry import download_weights
        with pytest.raises(ValueError, match="Unknown model"):
            download_weights("nonexistent_model")

    def test_get_config_unknown_model_raises(self):
        from molfun.hub.registry import get_config
        with pytest.raises(ValueError, match="Unknown model"):
            get_config("nonexistent_model")

    def test_download_uses_cache(self):
        from molfun.hub.registry import download_weights

        with tempfile.TemporaryDirectory() as tmp:
            weights_dir = Path(tmp) / "openfold"
            weights_dir.mkdir()
            fake_weights = weights_dir / "finetuning_ptm_2.pt"
            fake_weights.write_bytes(b"fake_weights_data")

            result = download_weights("openfold", cache_dir=tmp)
            assert result == fake_weights
            assert fake_weights.read_bytes() == b"fake_weights_data"

    def test_download_file_with_progress(self):
        from molfun.hub.registry import _download_file

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "test.bin"
            test_data = b"hello_world_test_data"

            mock_resp = MagicMock()
            mock_resp.headers = {"Content-Length": str(len(test_data))}
            mock_resp.read = MagicMock(side_effect=[test_data, b""])
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)

            with patch("urllib.request.urlopen", return_value=mock_resp):
                _download_file("https://example.com/file.pt", dest, progress=False)

            assert dest.exists()
            assert dest.read_bytes() == test_data

    def test_verify_sha256_correct(self):
        import hashlib
        from molfun.hub.registry import _verify_sha256

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.bin"
            p.write_bytes(b"test_data")
            expected = hashlib.sha256(b"test_data").hexdigest()
            _verify_sha256(p, expected)

    def test_verify_sha256_mismatch(self):
        from molfun.hub.registry import _verify_sha256

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.bin"
            p.write_bytes(b"test_data")
            with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
                _verify_sha256(p, "wrong_hash")


class TestFromPretrained:
    def test_from_pretrained_unknown_raises(self):
        from molfun.models import MolfunStructureModel
        with pytest.raises(ValueError, match="Unknown pretrained"):
            MolfunStructureModel.from_pretrained("nonexistent")

    def test_available_pretrained(self):
        from molfun.models import MolfunStructureModel
        names = MolfunStructureModel.available_pretrained()
        assert "openfold" in names

    @patch("molfun.hub.registry.download_weights")
    @patch("molfun.hub.registry.get_config")
    @patch("molfun.models.structure.MolfunStructureModel.__init__", return_value=None)
    def test_from_pretrained_calls_download(self, mock_init, mock_config, mock_download):
        from molfun.models import MolfunStructureModel

        mock_download.return_value = Path("/tmp/fake/weights.pt")
        mock_config.return_value = MagicMock()

        MolfunStructureModel.from_pretrained("openfold", device="cpu")

        mock_download.assert_called_once_with(
            "openfold", force=False, cache_dir=None,
        )
        mock_config.assert_called_once_with("openfold")
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args
        assert call_kwargs.kwargs["name"] == "openfold"
        assert call_kwargs.kwargs["device"] == "cpu"


class TestPredictSequence:
    def _make_model_with_mock_adapter(self):
        """Create a MolfunStructureModel with a mock adapter."""
        from molfun.models.structure import MolfunStructureModel
        from molfun.core.types import TrunkOutput

        model = MolfunStructureModel.__new__(MolfunStructureModel)
        model.name = "openfold"
        model.device = "cpu"
        model._peft = None
        model._strategy = None
        model.head = None

        mock_adapter = MagicMock()
        mock_adapter.eval = MagicMock()
        mock_output = TrunkOutput(
            single_repr=torch.randn(10, 256),
            pair_repr=torch.randn(10, 10, 128),
        )
        mock_adapter.return_value = mock_output
        model.adapter = mock_adapter

        return model, mock_output

    def test_predict_with_dict(self):
        model, expected = self._make_model_with_mock_adapter()
        batch = {"aatype": torch.zeros(10, dtype=torch.long)}
        result = model.predict(batch)
        model.adapter.assert_called_once_with(batch)

    @patch("molfun.models.structure.MolfunStructureModel._featurize_sequence")
    def test_predict_with_string(self, mock_feat):
        model, expected = self._make_model_with_mock_adapter()
        mock_feat.return_value = {"aatype": torch.zeros(5, dtype=torch.long)}

        result = model.predict("ACDEF")

        mock_feat.assert_called_once_with("ACDEF")
        model.adapter.assert_called_once()

    def test_featurize_unknown_backend_raises(self):
        from molfun.models.structure import MolfunStructureModel

        model = MolfunStructureModel.__new__(MolfunStructureModel)
        model.name = "unknown_backend"
        model.device = "cpu"

        class FakeAdapter:
            pass

        model.adapter = FakeAdapter()

        with pytest.raises(NotImplementedError, match="not implemented"):
            model._featurize_sequence("ACDEF")


class TestExampleDataset:
    def test_unknown_dataset_raises(self):
        from molfun.models import MolfunStructureModel
        with pytest.raises(ValueError, match="Unknown example"):
            MolfunStructureModel.example_dataset("nonexistent")

    def test_returns_cached_if_exists(self):
        from molfun.models import MolfunStructureModel

        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = Path(tmp) / "globins-small"
            ds_dir.mkdir()
            (ds_dir / "1mba.cif").write_text("fake")
            (ds_dir / "2hhb.cif").write_text("fake")

            paths = MolfunStructureModel.example_dataset(
                "globins-small", cache_dir=tmp,
            )
            assert len(paths) == 2
            assert all(p.endswith(".cif") for p in paths)

    @patch("molfun.data.collections.fetch_collection")
    def test_fetches_when_empty(self, mock_fetch):
        from molfun.models import MolfunStructureModel

        mock_fetch.return_value = ["/tmp/a.cif", "/tmp/b.cif"]

        with tempfile.TemporaryDirectory() as tmp:
            paths = MolfunStructureModel.example_dataset(
                "kinases-small", cache_dir=tmp,
            )
            assert paths == ["/tmp/a.cif", "/tmp/b.cif"]
            mock_fetch.assert_called_once()
            call_kwargs = mock_fetch.call_args
            assert call_kwargs.kwargs["max_structures"] == 30
