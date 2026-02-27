"""Tests for molfun.ml.features."""

import numpy as np
import pytest

from molfun.ml.features import (
    ProteinFeaturizer,
    AVAILABLE_FEATURES,
    DEFAULT_FEATURES,
    _aa_composition,
    _dipeptide_composition,
    _length,
    _molecular_weight,
    _charge_ph7,
    _isoelectric_point,
    _hydrophobicity_stats,
    _aromaticity,
    _ss_propensity,
    _sequence_entropy,
    _gravy,
    _tiny_small_aliphatic_aromatic_polar_charged,
)

SEQS = [
    "ACDEFGHIKLMNPQRSTVWY",
    "AAAAAAAAAA",
    "MTKILLFLGVAAAVLA",
    "GRKKRRQRRR",
]


class TestIndividualFeatures:
    def test_aa_composition_uniform(self):
        seq = "ACDEFGHIKLMNPQRSTVWY"
        result = _aa_composition(seq)
        assert result.shape == (20,)
        np.testing.assert_allclose(result, np.full(20, 0.05))

    def test_aa_composition_single(self):
        result = _aa_composition("AAAA")
        assert result.shape == (20,)
        assert result[0] == 1.0  # A is index 0
        assert result.sum() == pytest.approx(1.0)

    def test_dipeptide_shape(self):
        result = _dipeptide_composition("ACDEFG")
        assert result.shape == (400,)
        assert result.sum() == pytest.approx(1.0)

    def test_length(self):
        result = _length("ABCDE")
        assert result == [5]

    def test_molecular_weight_positive(self):
        result = _molecular_weight("ACDEFGHIKLMNPQRSTVWY")
        assert result[0] > 0

    def test_charge_ph7(self):
        neutral = _charge_ph7("AAAA")
        assert neutral[0] == pytest.approx(0.0)

        positive = _charge_ph7("KKKK")
        assert positive[0] > 0

        negative = _charge_ph7("DDDD")
        assert negative[0] < 0

    def test_isoelectric_point_range(self):
        pi = _isoelectric_point("ACDEFGHIKLMNPQRSTVWY")
        assert 0 < pi[0] < 14

    def test_hydrophobicity_stats_shape(self):
        result = _hydrophobicity_stats("AIVLMFWP")
        assert result.shape == (3,)

    def test_aromaticity(self):
        result = _aromaticity("FWYFWY")
        assert result[0] == pytest.approx(1.0)
        result = _aromaticity("AAAA")
        assert result[0] == pytest.approx(0.0)

    def test_ss_propensity_shape(self):
        result = _ss_propensity("ACDEF")
        assert result.shape == (3,)

    def test_sequence_entropy_zero(self):
        result = _sequence_entropy("AAAA")
        assert result[0] == pytest.approx(0.0)

    def test_sequence_entropy_positive(self):
        result = _sequence_entropy("ACDEFGHIKLMNPQRSTVWY")
        assert result[0] > 0

    def test_gravy(self):
        result = _gravy("AAAA")
        assert result.shape == (1,)

    def test_aa_groups_shape(self):
        result = _tiny_small_aliphatic_aromatic_polar_charged("ACDEFGHIKLMNPQRSTVWY")
        assert result.shape == (6,)


class TestProteinFeaturizer:
    def test_default_features(self):
        feat = ProteinFeaturizer()
        X = feat.fit_transform(SEQS)
        assert X.shape[0] == len(SEQS)
        assert X.shape[1] == feat.n_features

    def test_custom_features(self):
        feat = ProteinFeaturizer(features=["length", "charge_ph7"])
        X = feat.transform(SEQS)
        assert X.shape == (len(SEQS), 2)

    def test_all_features(self):
        feat = ProteinFeaturizer(features=AVAILABLE_FEATURES)
        X = feat.transform(SEQS)
        assert X.shape[0] == len(SEQS)
        assert X.shape[1] == feat.n_features

    def test_feature_names(self):
        feat = ProteinFeaturizer(features=["aa_composition", "length"])
        names = feat.feature_names
        assert len(names) == 21  # 20 + 1
        assert names[-1] == "length"

    def test_unknown_feature_raises(self):
        with pytest.raises(ValueError, match="Unknown feature"):
            ProteinFeaturizer(features=["does_not_exist"])

    def test_describe(self):
        feat = ProteinFeaturizer(features=["length", "gravy"])
        info = feat.describe()
        assert len(info) == 2
        assert info[0]["name"] == "length"
        assert info[0]["dims"] == 1

    def test_fit_is_noop(self):
        feat = ProteinFeaturizer()
        result = feat.fit(SEQS)
        assert result is feat

    def test_get_set_params(self):
        feat = ProteinFeaturizer(features=["length"])
        assert feat.get_params()["features"] == ["length"]
        feat.set_params(features=["gravy"])
        assert feat.features == ["gravy"]

    def test_empty_sequence(self):
        feat = ProteinFeaturizer(features=["length", "charge_ph7", "gravy"])
        X = feat.transform([""])
        assert X.shape == (1, 3)
        assert X[0, 0] == 0.0
