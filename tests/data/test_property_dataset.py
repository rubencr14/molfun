"""Tests for PropertyDataset."""

import numpy as np
import pytest
import os

from molfun.data.datasets.property import PropertyDataset


# ==================================================================
# Construction
# ==================================================================


class TestFromCSV:
    def _write_csv(self, tmp_path, header, rows, sep=","):
        path = tmp_path / "data.csv"
        with open(path, "w") as f:
            f.write(sep.join(header) + "\n")
            for row in rows:
                f.write(sep.join(str(x) for x in row) + "\n")
        return str(path)

    def test_basic_csv(self, tmp_path):
        path = self._write_csv(tmp_path,
            ["sequence", "target"],
            [["AAAA", 1.5], ["KKKK", 3.2], ["DDDD", 0.8]])
        ds = PropertyDataset.from_csv(path)
        assert len(ds) == 3
        assert ds.sequences == ["AAAA", "KKKK", "DDDD"]
        np.testing.assert_allclose(ds.targets, [1.5, 3.2, 0.8])
        assert ds.has_sequences
        assert ds.has_targets
        assert not ds.has_structures

    def test_csv_with_pdb_ids(self, tmp_path):
        path = self._write_csv(tmp_path,
            ["pdb_id", "sequence", "target"],
            [["1ABC", "AAAA", 1.0], ["2DEF", "KKKK", 2.0]])
        ds = PropertyDataset.from_csv(path)
        assert ds.pdb_ids == ["1ABC", "2DEF"]
        assert ds.sequences == ["AAAA", "KKKK"]

    def test_csv_with_pdb_dir(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        (pdb_dir / "1abc.cif").touch()
        (pdb_dir / "2def.cif").touch()

        path = self._write_csv(tmp_path,
            ["pdb_id", "target"],
            [["1abc", 5.0], ["2def", 6.0]])
        ds = PropertyDataset.from_csv(path, pdb_dir=str(pdb_dir))
        assert len(ds.pdb_paths) == 2
        assert ds.has_structures

    def test_csv_custom_columns(self, tmp_path):
        path = self._write_csv(tmp_path,
            ["seq", "kd"],
            [["AAAA", 100], ["KKKK", 200]])
        ds = PropertyDataset.from_csv(path, sequence_col="seq", target_col="kd")
        np.testing.assert_allclose(ds.targets, [100, 200])
        assert ds.target_name == "kd"

    def test_csv_tsv(self, tmp_path):
        path = self._write_csv(tmp_path,
            ["sequence", "target"],
            [["AAAA", 1.0]], sep="\t")
        ds = PropertyDataset.from_csv(path)
        assert len(ds) == 1

    def test_csv_missing_file(self):
        with pytest.raises(FileNotFoundError):
            PropertyDataset.from_csv("/nonexistent.csv")

    def test_csv_empty(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("sequence,target\n")
        with pytest.raises(ValueError, match="Empty"):
            PropertyDataset.from_csv(str(path))

    def test_csv_no_target_column(self, tmp_path):
        path = self._write_csv(tmp_path,
            ["sequence", "something_else"],
            [["AAAA", "foo"]])
        ds = PropertyDataset.from_csv(path)
        assert not ds.has_targets
        assert ds.sequences == ["AAAA"]

    def test_csv_metadata(self, tmp_path):
        path = self._write_csv(tmp_path,
            ["sequence", "target", "organism"],
            [["AAAA", 1.0, "human"]])
        ds = PropertyDataset.from_csv(path)
        assert ds.metadata[0]["organism"] == "human"


class TestFromInline:
    def test_sequences_and_targets(self):
        ds = PropertyDataset.from_inline(
            sequences=["AAAA", "KKKK"],
            targets=[1.0, 2.0],
        )
        assert len(ds) == 2
        assert ds.has_sequences
        assert ds.has_targets

    def test_empty(self):
        ds = PropertyDataset.from_inline()
        assert len(ds) == 0


class TestFromFetchAndCSV:
    def test_basic_match(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text("pdb_id,target\n1ABC,5.0\n2DEF,6.0\n3GHI,7.0\n")

        pdb_paths = [
            tmp_path / "2DEF.cif",
            tmp_path / "1ABC.cif",
            tmp_path / "9ZZZ.cif",  # not in CSV
        ]
        for p in pdb_paths:
            p.touch()

        ds = PropertyDataset.from_fetch_and_csv(
            pdb_paths=[str(p) for p in pdb_paths],
            csv_path=str(csv_path),
        )
        assert len(ds) == 2
        assert set(ds.pdb_ids) == {"2DEF", "1ABC"}


# ==================================================================
# Methods
# ==================================================================


class TestSplit:
    def test_split(self):
        ds = PropertyDataset(
            sequences=["A", "K", "D", "V", "F", "S", "L", "E", "G", "R"],
            targets=np.arange(10, dtype=np.float64),
        )
        train, test = ds.split(test_frac=0.3, seed=42)
        assert len(train) + len(test) == 10
        assert len(test) == 3
        assert train.has_targets
        assert test.has_targets


class TestToDict:
    def test_to_dict(self):
        ds = PropertyDataset(
            sequences=["AAAA", "KKKK"],
            targets=np.array([1.0, 2.0]),
            pdb_ids=["1ABC", "2DEF"],
        )
        d = ds.to_dict()
        assert d["sequences"] == ["AAAA", "KKKK"]
        np.testing.assert_allclose(d["y"], [1.0, 2.0])
        assert d["pdb_ids"] == ["1ABC", "2DEF"]


class TestRepr:
    def test_repr(self):
        ds = PropertyDataset(
            sequences=["AA", "KK"],
            targets=np.array([1.0, 2.0]),
        )
        r = repr(ds)
        assert "n=2" in r
        assert "sequences=2" in r
        assert "targets=2" in r
