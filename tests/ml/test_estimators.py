"""Tests for molfun.ml.estimators."""

import numpy as np
import pytest
import tempfile
import os

from molfun.ml.features import ProteinFeaturizer
from molfun.ml.estimators import (
    ProteinClassifier,
    ProteinRegressor,
    CLASSIFIER_NAMES,
    REGRESSOR_NAMES,
)


TRAIN_SEQS = [
    "AAAAAAAAA", "KKKKKKKKK", "DDDDDDDDD", "VVVVVVVVV",
    "FFFFFFFFF", "SSSSSSSSSS", "LLLLLLLLL", "EEEEEEEEE",
    "GGGGGGGGG", "RRRRRRRRR", "IIIIIIIII", "NNNNNNNNN",
    "IIIIIIIII", "WWWWWWWWW", "TTTTTTTTT", "PPPPPPPPP",
]

TRAIN_Y_REG = np.random.RandomState(42).rand(len(TRAIN_SEQS))
TRAIN_Y_CLF = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

TEST_SEQS = ["AAAVVV", "KKKDDD", "FFLLEE", "GGRRNN"]
TEST_Y_REG = np.random.RandomState(99).rand(len(TEST_SEQS))
TEST_Y_CLF = np.array([0, 1, 0, 1])


class TestProteinRegressor:
    def test_fit_predict_default(self):
        reg = ProteinRegressor(estimator="random_forest", n_estimators=10)
        reg.fit(TRAIN_SEQS, TRAIN_Y_REG)
        preds = reg.predict(TEST_SEQS)
        assert len(preds) == len(TEST_SEQS)

    def test_score(self):
        reg = ProteinRegressor(estimator="random_forest", n_estimators=10)
        reg.fit(TRAIN_SEQS, TRAIN_Y_REG)
        score = reg.score(TEST_SEQS, TEST_Y_REG)
        assert isinstance(score, float)

    def test_evaluate(self):
        reg = ProteinRegressor(estimator="random_forest", n_estimators=10)
        reg.fit(TRAIN_SEQS, TRAIN_Y_REG)
        metrics = reg.evaluate(TEST_SEQS, TEST_Y_REG)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "pearson" in metrics
        assert metrics["n_samples"] == len(TEST_SEQS)

    def test_top_features(self):
        reg = ProteinRegressor(estimator="random_forest", n_estimators=10)
        reg.fit(TRAIN_SEQS, TRAIN_Y_REG)
        top = reg.top_features(5)
        assert len(top) <= 5
        for name, imp in top:
            assert isinstance(name, str)
            assert imp >= 0

    def test_describe(self):
        reg = ProteinRegressor(estimator="random_forest", n_estimators=10)
        desc = reg.describe()
        assert desc["estimator"] == "random_forest"
        assert desc["fitted"] is False

    def test_not_fitted_raises(self):
        reg = ProteinRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            reg.predict(TEST_SEQS)

    def test_custom_featurizer(self):
        feat = ProteinFeaturizer(features=["aa_composition", "length"])
        reg = ProteinRegressor(estimator="random_forest", featurizer=feat, n_estimators=10)
        reg.fit(TRAIN_SEQS, TRAIN_Y_REG)
        preds = reg.predict(TEST_SEQS)
        assert len(preds) == len(TEST_SEQS)

    def test_featurizer_and_features_conflict(self):
        feat = ProteinFeaturizer(features=["length"])
        with pytest.raises(ValueError, match="either"):
            ProteinRegressor(featurizer=feat, features=["gravy"])

    def test_linear_regressor(self):
        reg = ProteinRegressor(estimator="linear")
        reg.fit(TRAIN_SEQS, TRAIN_Y_REG)
        preds = reg.predict(TEST_SEQS)
        assert len(preds) == len(TEST_SEQS)

    def test_svm_regressor(self):
        reg = ProteinRegressor(estimator="svm")
        reg.fit(TRAIN_SEQS, TRAIN_Y_REG)
        preds = reg.predict(TEST_SEQS)
        assert len(preds) == len(TEST_SEQS)


class TestProteinClassifier:
    def test_fit_predict(self):
        clf = ProteinClassifier(estimator="random_forest", n_estimators=10)
        clf.fit(TRAIN_SEQS, TRAIN_Y_CLF)
        preds = clf.predict(TEST_SEQS)
        assert len(preds) == len(TEST_SEQS)

    def test_predict_proba(self):
        clf = ProteinClassifier(estimator="random_forest", n_estimators=10)
        clf.fit(TRAIN_SEQS, TRAIN_Y_CLF)
        proba = clf.predict_proba(TEST_SEQS)
        assert proba.shape == (len(TEST_SEQS), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_svm_classifier(self):
        clf = ProteinClassifier(estimator="svm")
        clf.fit(TRAIN_SEQS, TRAIN_Y_CLF)
        preds = clf.predict(TEST_SEQS)
        assert len(preds) == len(TEST_SEQS)

    def test_logistic_classifier(self):
        clf = ProteinClassifier(estimator="logistic")
        clf.fit(TRAIN_SEQS, TRAIN_Y_CLF)
        preds = clf.predict(TEST_SEQS)
        assert len(preds) == len(TEST_SEQS)


class TestSaveLoad:
    def test_save_load_roundtrip(self):
        reg = ProteinRegressor(estimator="random_forest", n_estimators=10)
        reg.fit(TRAIN_SEQS, TRAIN_Y_REG)
        preds_before = reg.predict(TEST_SEQS)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            reg.save(path)

            loaded = ProteinRegressor.load(path)
            preds_after = loaded.predict(TEST_SEQS)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            ProteinRegressor.load("/tmp/nonexistent_model_xyz.joblib")


class TestGetParams:
    def test_regressor_params(self):
        reg = ProteinRegressor(estimator="random_forest", n_estimators=10)
        params = reg.get_params()
        assert params["estimator"] == "random_forest"
        assert params["n_estimators"] == 10

    def test_unknown_estimator_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            reg = ProteinRegressor(estimator="does_not_exist")
            reg.fit(TRAIN_SEQS, TRAIN_Y_REG)

    def test_unknown_classifier_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            clf = ProteinClassifier(estimator="does_not_exist")
            clf.fit(TRAIN_SEQS, TRAIN_Y_CLF)
