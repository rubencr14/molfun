"""Tests for the high-level prediction API (molfun/predict.py)."""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from molfun.core.types import TrunkOutput


def _mock_trunk_output(L=10, D=256):
    return TrunkOutput(
        single_repr=torch.randn(1, L, D),
        pair_repr=torch.randn(1, L, L, 128),
        structure_coords=torch.randn(1, L, 37, 3),
        confidence=torch.rand(1, L),
    )


class TestPredictStructure:
    @patch("molfun.predict._get_model")
    def test_returns_expected_keys(self, mock_get):
        model = MagicMock()
        model.predict.return_value = _mock_trunk_output(L=5)
        mock_get.return_value = model

        from molfun.predict import predict_structure
        result = predict_structure("ACDEF", device="cpu")

        assert "coordinates" in result
        assert "plddt" in result
        assert "pdb_string" in result
        assert "sequence" in result
        assert "length" in result
        assert result["sequence"] == "ACDEF"
        assert result["length"] == 5

    @patch("molfun.predict._get_model")
    def test_coordinates_shape(self, mock_get):
        model = MagicMock()
        model.predict.return_value = _mock_trunk_output(L=8)
        mock_get.return_value = model

        from molfun.predict import predict_structure
        result = predict_structure("ACDEFGHI", device="cpu")

        assert len(result["coordinates"]) == 8
        assert len(result["coordinates"][0]) == 3

    @patch("molfun.predict._get_model")
    def test_plddt_length_matches(self, mock_get):
        model = MagicMock()
        model.predict.return_value = _mock_trunk_output(L=6)
        mock_get.return_value = model

        from molfun.predict import predict_structure
        result = predict_structure("ACDEFG", device="cpu")

        assert len(result["plddt"]) == 6
        assert all(isinstance(v, float) for v in result["plddt"])

    @patch("molfun.predict._get_model")
    def test_pdb_string_has_atoms(self, mock_get):
        model = MagicMock()
        model.predict.return_value = _mock_trunk_output(L=3)
        mock_get.return_value = model

        from molfun.predict import predict_structure
        result = predict_structure("ACE", device="cpu")

        assert "ATOM" in result["pdb_string"]
        assert "END" in result["pdb_string"]
        assert "ALA" in result["pdb_string"]
        assert "CYS" in result["pdb_string"]
        assert "GLU" in result["pdb_string"]

    @patch("molfun.predict._get_model")
    def test_handles_no_coords(self, mock_get):
        model = MagicMock()
        output = TrunkOutput(single_repr=torch.randn(1, 5, 256))
        model.predict.return_value = output
        mock_get.return_value = model

        from molfun.predict import predict_structure
        result = predict_structure("ACDEF", device="cpu")

        assert result["coordinates"] == []
        assert result["pdb_string"] == ""


class TestPredictProperties:
    def test_sequence_properties_no_model(self):
        from molfun.predict import predict_properties

        result = predict_properties("MKWVTFISLLLLFSSAYS")

        assert "molecular_weight" in result
        assert "hydrophobicity" in result
        assert "charge" in result
        assert "aromaticity" in result
        assert "isoelectric_point" in result
        assert "instability_index" in result

    def test_molecular_weight_reasonable(self):
        from molfun.predict import predict_properties

        result = predict_properties("A", properties=["molecular_weight"])
        assert 80 < result["molecular_weight"] < 100

        result = predict_properties("AAAAAAAAAA", properties=["molecular_weight"])
        assert 700 < result["molecular_weight"] < 800

    def test_hydrophobicity_range(self):
        from molfun.predict import predict_properties

        result = predict_properties("IIIVVVLLL", properties=["hydrophobicity"])
        assert result["hydrophobicity"] > 2.0

        result = predict_properties("DDDEEERRRK", properties=["hydrophobicity"])
        assert result["hydrophobicity"] < 0.0

    def test_charge(self):
        from molfun.predict import predict_properties

        result = predict_properties("KKKRRR", properties=["charge"])
        assert result["charge"] > 0

        result = predict_properties("DDDEEE", properties=["charge"])
        assert result["charge"] < 0

    def test_aromaticity(self):
        from molfun.predict import predict_properties

        result = predict_properties("FWYFWY", properties=["aromaticity"])
        assert result["aromaticity"] > 0.8

        result = predict_properties("AAAAAA", properties=["aromaticity"])
        assert result["aromaticity"] == 0.0

    def test_isoelectric_point_range(self):
        from molfun.predict import predict_properties

        result = predict_properties("MKWVTFISLLLLFSSAYS", properties=["isoelectric_point"])
        assert 3.0 < result["isoelectric_point"] < 12.0

    def test_unknown_property_raises(self):
        from molfun.predict import predict_properties

        with pytest.raises(ValueError, match="Unknown properties"):
            predict_properties("ACDEF", properties=["nonexistent"])

    @patch("molfun.predict._get_model")
    def test_embedding_properties(self, mock_get):
        model = MagicMock()
        model.predict.return_value = _mock_trunk_output(L=5)
        mock_get.return_value = model

        from molfun.predict import predict_properties
        result = predict_properties(
            "ACDEF",
            properties=["stability", "solubility"],
            device="cpu",
        )

        assert "stability" in result
        assert "solubility" in result
        assert 0.0 <= result["stability"] <= 1.0
        assert 0.0 <= result["solubility"] <= 1.0


class TestPredictAffinity:
    @patch("molfun.predict._get_model")
    def test_returns_expected_keys(self, mock_get):
        model = MagicMock()
        model.predict.return_value = _mock_trunk_output(L=5)
        model.forward.return_value = {"preds": torch.tensor([[- 7.5]])}
        model._featurize_sequence.return_value = {"aatype": torch.zeros(5)}
        mock_get.return_value = model

        from molfun.predict import predict_affinity
        result = predict_affinity("ACDEF", ligand_smiles="CC(=O)O", device="cpu")

        assert "binding_affinity_kcal" in result
        assert "confidence" in result
        assert "ligand_smiles" in result
        assert "sequence_length" in result
        assert result["ligand_smiles"] == "CC(=O)O"
        assert result["sequence_length"] == 5

    @patch("molfun.predict._get_model")
    def test_affinity_value(self, mock_get):
        model = MagicMock()
        model.predict.return_value = _mock_trunk_output(L=5)
        model.forward.return_value = {"preds": torch.tensor([[-8.3]])}
        model._featurize_sequence.return_value = {"aatype": torch.zeros(5)}
        mock_get.return_value = model

        from molfun.predict import predict_affinity
        result = predict_affinity("ACDEF", ligand_smiles="c1ccccc1", device="cpu")

        assert abs(result["binding_affinity_kcal"] - (-8.3)) < 0.01


class TestModelCache:
    def test_clear_cache(self):
        from molfun.predict import _MODEL_CACHE, clear_cache

        _MODEL_CACHE["test:cpu:None"] = "dummy"
        assert len(_MODEL_CACHE) > 0

        clear_cache()
        assert len(_MODEL_CACHE) == 0


class TestHelpers:
    def test_coords_to_pdb(self):
        from molfun.predict import _coords_to_pdb

        coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        pdb = _coords_to_pdb("ACE", coords)

        lines = pdb.strip().split("\n")
        assert len(lines) == 4
        assert lines[0].startswith("ATOM")
        assert "ALA" in lines[0]
        assert "CYS" in lines[1]
        assert "GLU" in lines[2]
        assert lines[3] == "END"

    def test_sigmoid(self):
        from molfun.predict import _sigmoid

        assert abs(_sigmoid(0) - 0.5) < 1e-6
        assert _sigmoid(100) > 0.999
        assert _sigmoid(-100) < 0.001

    def test_compute_pi(self):
        from molfun.predict import _compute_pi

        pi = _compute_pi("MKWVTFISLLLLFSSAYS")
        assert 3.0 < pi < 12.0

        pi_acidic = _compute_pi("DDDEEEE")
        pi_basic = _compute_pi("KKKKRRR")
        assert pi_acidic < pi_basic

    def test_available_properties(self):
        from molfun.predict import AVAILABLE_PROPERTIES

        assert "molecular_weight" in AVAILABLE_PROPERTIES
        assert "stability" in AVAILABLE_PROPERTIES
        assert len(AVAILABLE_PROPERTIES) >= 10
