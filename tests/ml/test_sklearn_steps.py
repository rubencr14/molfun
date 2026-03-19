"""Tests for sklearn pipeline steps in molfun.pipelines.steps."""

import numpy as np
import pytest

from molfun.pipelines.steps import (
    featurize_step,
    split_sklearn_step,
    train_sklearn_step,
    eval_sklearn_step,
    save_sklearn_step,
)


SEQS = [
    "AAAAAAAAA", "KKKKKKKKK", "DDDDDDDDD", "VVVVVVVVV",
    "FFFFFFFFF", "SSSSSSSSSS", "LLLLLLLLL", "EEEEEEEEE",
    "GGGGGGGGG", "RRRRRRRRR", "IIIIIIIII", "NNNNNNNNN",
    "IIIIIIIII", "WWWWWWWWW", "TTTTTTTTT", "PPPPPPPPP",
    "AAAKKKDDD", "VVVFFFSSS", "LLLEEEGGG", "RRRIIIYNN",
]

TARGETS = np.random.RandomState(42).rand(len(SEQS))
LABELS = np.array([0, 1] * 10)


class TestFeaturizeStep:
    def test_basic(self):
        state = {"sequences": SEQS, "features": ["aa_composition", "length"]}
        result = featurize_step(state)
        assert "X" in result
        assert result["X"].shape[0] == len(SEQS)
        assert result["X"].shape[1] == 21  # 20 + 1
        assert "feature_names" in result

    def test_default_features(self):
        state = {"sequences": SEQS}
        result = featurize_step(state)
        assert "X" in result
        assert result["X"].shape[0] == len(SEQS)

    def test_missing_data_raises(self):
        with pytest.raises(ValueError, match="sequences.*pdb_paths"):
            featurize_step({})


class TestSplitSklearnStep:
    def test_basic_split(self):
        X = np.random.rand(20, 5)
        state = {"X": X, "y": TARGETS, "val_frac": 0.2, "seed": 42}
        result = split_sklearn_step(state)
        assert "X_train" in result
        assert "X_test" in result
        assert "y_train" in result
        assert "y_test" in result
        assert len(result["X_train"]) + len(result["X_test"]) == 20

    def test_with_sequences(self):
        X = np.random.rand(20, 5)
        state = {"X": X, "y": TARGETS, "sequences": SEQS, "val_frac": 0.2}
        result = split_sklearn_step(state)
        assert "sequences_train" in result
        assert "sequences_test" in result


class TestTrainSklearnStep:
    def _make_state(self, task="regression"):
        state = {"sequences": SEQS, "features": ["aa_composition", "length"]}
        state = featurize_step(state)
        state["y"] = TARGETS if task == "regression" else LABELS
        state = split_sklearn_step(state)
        return state

    def test_regression_random_forest(self):
        state = self._make_state("regression")
        state["estimator"] = "random_forest"
        state["task"] = "regression"
        state["n_estimators"] = 10
        result = train_sklearn_step(state)
        assert "model" in result

    def test_classification_svm(self):
        state = self._make_state("classification")
        state["estimator"] = "svm"
        state["task"] = "classification"
        result = train_sklearn_step(state)
        assert "model" in result

    def test_missing_data_raises(self):
        state = {"y_train": [1, 2, 3], "task": "regression"}
        with pytest.raises(ValueError, match="X_train.*sequences_train"):
            train_sklearn_step(state)


class TestEvalSklearnStep:
    def test_eval_regression(self):
        state = {"sequences": SEQS, "features": ["aa_composition", "length"]}
        state = featurize_step(state)
        state["y"] = TARGETS
        state = split_sklearn_step(state)
        state["estimator"] = "random_forest"
        state["task"] = "regression"
        state["n_estimators"] = 10
        state = train_sklearn_step(state)
        result = eval_sklearn_step(state)
        assert "eval_metrics" in result
        assert "mae" in result["eval_metrics"]

    def test_missing_data_raises(self):
        from molfun.ml import ProteinRegressor
        model = ProteinRegressor(estimator="random_forest", n_estimators=10)
        model.fit(SEQS, TARGETS)
        state = {"model": model, "y_test": [1, 2]}
        with pytest.raises(ValueError, match="X_test.*sequences_test"):
            eval_sklearn_step(state)


class TestSaveSklearnStep:
    def test_save(self, tmp_path):
        state = {"sequences": SEQS, "features": ["length"]}
        state = featurize_step(state)
        state["y"] = TARGETS
        state = split_sklearn_step(state)
        state["estimator"] = "random_forest"
        state["task"] = "regression"
        state["n_estimators"] = 10
        state = train_sklearn_step(state)

        model_path = str(tmp_path / "model.joblib")
        state["model_path"] = model_path
        result = save_sklearn_step(state)
        assert result["model_path"] == model_path
        assert (tmp_path / "model.joblib").exists()
