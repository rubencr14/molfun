"""Tests for load_dataset_step and PropertyHead."""

import csv
import numpy as np
import pytest
import os

from molfun.pipelines.steps import load_dataset_step, load_targets_step
from molfun.ml.heads import PropertyHead


# ==================================================================
# load_dataset_step
# ==================================================================


class TestLoadDatasetCSV:
    def _write_csv(self, tmpdir, rows, sep=","):
        path = os.path.join(tmpdir, "targets.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter=sep)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_basic_csv(self, tmp_path):
        rows = [
            {"sequence": "AAAA", "target": "1.5"},
            {"sequence": "KKKK", "target": "3.2"},
            {"sequence": "DDDD", "target": "0.8"},
        ]
        path = self._write_csv(str(tmp_path), rows)
        state = {"source": "csv", "targets_file": path}
        result = load_dataset_step(state)
        assert "y" in result
        assert len(result["y"]) == 3
        np.testing.assert_allclose(result["y"], [1.5, 3.2, 0.8])
        assert result["sequences"] == ["AAAA", "KKKK", "DDDD"]
        assert "dataset" in result

    def test_csv_custom_columns(self, tmp_path):
        rows = [
            {"seq": "AAAA", "kd": "100"},
            {"seq": "KKKK", "kd": "200"},
        ]
        path = self._write_csv(str(tmp_path), rows)
        state = {
            "source": "csv",
            "targets_file": path,
            "sequence_col": "seq",
            "target_col": "kd",
        }
        result = load_dataset_step(state)
        np.testing.assert_allclose(result["y"], [100.0, 200.0])

    def test_csv_tsv_autodetect(self, tmp_path):
        path = os.path.join(str(tmp_path), "data.tsv")
        with open(path, "w") as f:
            f.write("sequence\ttarget\n")
            f.write("AAAA\t1.0\n")
            f.write("KKKK\t2.0\n")
        state = {"source": "csv", "targets_file": path}
        result = load_dataset_step(state)
        np.testing.assert_allclose(result["y"], [1.0, 2.0])

    def test_csv_missing_file_raises(self):
        state = {"source": "csv", "targets_file": "/nonexistent/file.csv"}
        with pytest.raises(FileNotFoundError):
            load_dataset_step(state)


class TestLoadDatasetInline:
    def test_basic_inline(self):
        state = {"source": "inline", "targets": [1.0, 2.0, 3.0]}
        result = load_dataset_step(state)
        np.testing.assert_allclose(result["y"], [1.0, 2.0, 3.0])

    def test_inline_with_sequences(self):
        state = {
            "source": "inline",
            "targets": [1.0, 2.0],
            "sequences": ["AAAA", "KKKK"],
        }
        result = load_dataset_step(state)
        assert result["sequences"] == ["AAAA", "KKKK"]


class TestLoadDatasetFetchCSV:
    def test_fetch_csv(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text("pdb_id,target\n1ABC,5.0\n2DEF,6.0\n")

        pdb_paths = [str(tmp_path / "1ABC.cif"), str(tmp_path / "2DEF.cif")]
        for p in pdb_paths:
            open(p, "w").close()

        state = {
            "source": "fetch_csv",
            "targets_file": str(csv_path),
            "pdb_paths": pdb_paths,
        }
        result = load_dataset_step(state)
        assert "y" in result
        assert len(result["y"]) == 2

    def test_fetch_csv_no_paths_raises(self):
        with pytest.raises(ValueError, match="pdb_paths"):
            load_dataset_step({"source": "fetch_csv", "targets_file": "x.csv"})


class TestUnknownSource:
    def test_unknown_source_raises(self):
        state = {"source": "database_xyz"}
        with pytest.raises(ValueError, match="Unknown"):
            load_dataset_step(state)


class TestBackwardAlias:
    def test_load_targets_step_is_alias(self):
        assert load_targets_step is load_dataset_step


# ==================================================================
# PropertyHead (sklearn heads only, since torch may not be available)
# ==================================================================


class TestPropertyHeadSklearn:
    def _make_data(self, n=50, dim=32):
        rng = np.random.RandomState(42)
        X = rng.randn(n, dim)
        y = X[:, 0] * 2 + X[:, 1] + rng.randn(n) * 0.1
        return X, y

    def test_linear_regression_head(self):
        X, y = self._make_data()
        head = PropertyHead(head_type="linear", task="regression")
        head.fit(pdb_paths=[], y=y, embeddings=X)
        preds = head.predict(embeddings=X)
        assert len(preds) == len(y)
        assert head._fitted

    def test_rf_regression_head(self):
        X, y = self._make_data()
        head = PropertyHead(head_type="rf", task="regression", n_estimators=10)
        head.fit(pdb_paths=[], y=y, embeddings=X)
        preds = head.predict(embeddings=X)
        assert len(preds) == len(y)

    def test_svm_classification_head(self):
        X, _ = self._make_data()
        y = np.array([0, 1] * 25)
        head = PropertyHead(head_type="svm", task="classification")
        head.fit(pdb_paths=[], y=y, embeddings=X)
        preds = head.predict(embeddings=X)
        assert set(preds).issubset({0, 1})

    def test_not_fitted_raises(self):
        head = PropertyHead(head_type="linear")
        with pytest.raises(RuntimeError, match="not fitted"):
            head.predict(embeddings=np.zeros((2, 10)))

    def test_predict_no_data_raises(self):
        X, y = self._make_data()
        head = PropertyHead(head_type="linear", task="regression")
        head.fit(pdb_paths=[], y=y, embeddings=X)
        with pytest.raises(ValueError, match="pdb_paths or embeddings"):
            head.predict()

    def test_describe(self):
        head = PropertyHead(head_type="rf", task="regression")
        desc = head.describe()
        assert desc["head_type"] == "rf"
        assert desc["fitted"] is False

    def test_save_load(self, tmp_path):
        X, y = self._make_data()
        head = PropertyHead(head_type="linear", task="regression")
        head.fit(pdb_paths=[], y=y, embeddings=X)
        preds_before = head.predict(embeddings=X)

        path = str(tmp_path / "head.joblib")
        head.save(path)

        loaded = PropertyHead.load(path)
        preds_after = loaded.predict(embeddings=X)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_unknown_head_raises(self):
        X, y = self._make_data(n=10)
        head = PropertyHead(head_type="transformer")
        with pytest.raises(ValueError, match="Unknown head_type"):
            head.fit(pdb_paths=[], y=y, embeddings=X)

    def test_unknown_backbone_raises(self):
        from molfun.ml.heads import extract_embeddings
        with pytest.raises(ValueError, match="Unknown backbone"):
            extract_embeddings(["fake.pdb"], backbone="nonexistent_model")
