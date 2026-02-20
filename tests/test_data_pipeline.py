"""
Tests for the data pipeline: sources, datasets, splits.
Uses mock/local data to avoid network calls.
"""

import pytest
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from molfun.data.sources.pdb import PDBFetcher
from molfun.data.sources.affinity import AffinityFetcher, AffinityRecord
from molfun.data.datasets.structure import StructureDataset, collate_structure_batch
from molfun.data.datasets.affinity import AffinityDataset
from molfun.data.splits import DataSplitter


# ── Fixtures ──────────────────────────────────────────────────────────

SAMPLE_PDB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.500   2.500   3.500  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   3.000   4.000  1.00  0.00           C
ATOM      4  N   GLY A   2       3.000   4.000   5.000  1.00  0.00           N
ATOM      5  CA  GLY A   2       3.500   4.500   5.500  1.00  0.00           C
ATOM      6  C   GLY A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM      7  N   VAL A   3       5.000   6.000   7.000  1.00  0.00           N
ATOM      8  CA  VAL A   3       5.500   6.500   7.500  1.00  0.00           C
ATOM      9  C   VAL A   3       6.000   7.000   8.000  1.00  0.00           C
END
"""

SAMPLE_PDBBIND_INDEX = """\
# PDBbind v2020 refined set
1abc  1.80  2018  6.50  Kd=3.2e-7M  reference // (LIG)
2xyz  2.10  2019  7.20  Ki=6.3e-8M  reference // (ATP)
3def  1.50  2020  5.80  IC50=1.5e-6M  reference // (INH)
"""


@pytest.fixture
def pdb_dir(tmp_path):
    """Create temp directory with sample PDB files."""
    for name in ["1abc", "2xyz", "3def"]:
        (tmp_path / f"{name}.pdb").write_text(SAMPLE_PDB)
    return tmp_path


@pytest.fixture
def pdbbind_index(tmp_path):
    """Create temp PDBbind index file."""
    path = tmp_path / "INDEX_refined_data.2020"
    path.write_text(SAMPLE_PDBBIND_INDEX)
    return path


@pytest.fixture
def csv_file(tmp_path):
    """Create temp CSV affinity file."""
    path = tmp_path / "affinity.csv"
    path.write_text(
        "pdb_id,affinity,resolution\n"
        "1abc,6.5,1.8\n"
        "2xyz,7.2,2.1\n"
        "3def,5.8,1.5\n"
    )
    return path


# ── PDBFetcher ────────────────────────────────────────────────────────

class TestPDBFetcher:

    def test_init_creates_cache(self, tmp_path):
        cache = tmp_path / "cache"
        fetcher = PDBFetcher(cache_dir=str(cache))
        assert cache.exists()

    def test_list_cached_empty(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path))
        assert fetcher.list_cached() == []

    def test_invalid_fmt_raises(self, tmp_path):
        with pytest.raises(ValueError, match="fmt must be"):
            PDBFetcher(cache_dir=str(tmp_path), fmt="xyz")


# ── AffinityFetcher ──────────────────────────────────────────────────

class TestAffinityFetcher:

    def test_parse_pdbbind_index(self, pdbbind_index):
        records = AffinityFetcher.from_pdbbind_index(str(pdbbind_index))
        assert len(records) == 3
        assert records[0].pdb_id == "1abc"
        assert records[0].affinity == 6.5
        assert records[0].year == 2018
        assert records[0].resolution == 1.8
        assert records[1].pdb_id == "2xyz"
        assert records[2].ligand_name == "INH"

    def test_from_csv(self, csv_file):
        records = AffinityFetcher.from_csv(str(csv_file))
        assert len(records) == 3
        assert records[0].pdb_id == "1abc"
        assert records[1].affinity == 7.2

    def test_filter_resolution(self, pdbbind_index):
        records = AffinityFetcher.from_pdbbind_index(str(pdbbind_index))
        filtered = AffinityFetcher.filter(records, resolution_max=2.0)
        assert len(filtered) == 2
        assert all(r.resolution <= 2.0 for r in filtered)

    def test_filter_year(self, pdbbind_index):
        records = AffinityFetcher.from_pdbbind_index(str(pdbbind_index))
        filtered = AffinityFetcher.filter(records, min_year=2019)
        assert len(filtered) == 2

    def test_to_label_dict(self, pdbbind_index):
        records = AffinityFetcher.from_pdbbind_index(str(pdbbind_index))
        labels = AffinityFetcher.to_label_dict(records)
        assert labels["1abc"] == 6.5
        assert labels["2xyz"] == 7.2

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            AffinityFetcher.from_pdbbind_index("/nonexistent/file")


# ── StructureDataset ─────────────────────────────────────────────────

class TestStructureDataset:

    def test_load_pdb_files(self, pdb_dir):
        paths = list(pdb_dir.glob("*.pdb"))
        ds = StructureDataset(pdb_paths=paths)
        assert len(ds) == 3

    def test_getitem_returns_features_and_label(self, pdb_dir):
        paths = list(pdb_dir.glob("*.pdb"))
        ds = StructureDataset(pdb_paths=paths, labels={"1abc": 6.5})
        feat, label = ds[0]
        assert "sequence" in feat
        assert "all_atom_positions" in feat
        assert isinstance(label, torch.Tensor)

    def test_sequence_extraction(self, pdb_dir):
        paths = [pdb_dir / "1abc.pdb"]
        ds = StructureDataset(pdb_paths=paths)
        feat, _ = ds[0]
        assert feat["sequence"] == "AGV"
        assert feat["all_atom_positions"].shape == (3, 3)
        assert feat["all_atom_mask"].sum() == 3  # all CA present

    def test_pdb_ids(self, pdb_dir):
        paths = list(pdb_dir.glob("*.pdb"))
        ds = StructureDataset(pdb_paths=paths)
        assert set(ds.pdb_ids) == {"1abc", "2xyz", "3def"}

    def test_max_seq_len_crops(self, pdb_dir):
        paths = [pdb_dir / "1abc.pdb"]
        ds = StructureDataset(pdb_paths=paths, max_seq_len=2)
        feat, _ = ds[0]
        assert len(feat["sequence"]) == 2

    def test_precomputed_features(self, pdb_dir, tmp_path):
        import pickle
        feat_dir = tmp_path / "features"
        feat_dir.mkdir()
        fake_feat = {"sequence": "MOCK", "custom_key": torch.randn(4, 64)}
        with open(feat_dir / "1abc.pkl", "wb") as f:
            pickle.dump(fake_feat, f)

        ds = StructureDataset(
            pdb_paths=[pdb_dir / "1abc.pdb"],
            features_dir=str(feat_dir),
        )
        feat, _ = ds[0]
        assert feat["sequence"] == "MOCK"

    def test_collate_fn(self, pdb_dir):
        paths = list(pdb_dir.glob("*.pdb"))
        ds = StructureDataset(pdb_paths=paths, labels={"1abc": 6.5, "2xyz": 7.2, "3def": 5.8})
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_structure_batch)
        batch_feat, batch_labels = next(iter(loader))
        assert batch_feat["all_atom_positions"].shape[0] == 2
        assert batch_labels.shape[0] == 2


# ── AffinityDataset ──────────────────────────────────────────────────

class TestAffinityDataset:

    def test_from_records(self, pdb_dir, pdbbind_index):
        records = AffinityFetcher.from_pdbbind_index(str(pdbbind_index))
        ds = AffinityDataset.from_records(records, pdb_dir=pdb_dir, fmt="pdb")
        assert len(ds) == 3
        feat, label = ds[0]
        assert "sequence" in feat
        assert not torch.isnan(label)

    def test_from_csv(self, pdb_dir, csv_file):
        ds = AffinityDataset.from_csv(csv_path=csv_file, pdb_dir=pdb_dir, fmt="pdb")
        assert len(ds) == 3

    def test_labels_dict(self, pdb_dir, pdbbind_index):
        records = AffinityFetcher.from_pdbbind_index(str(pdbbind_index))
        ds = AffinityDataset.from_records(records, pdb_dir=pdb_dir, fmt="pdb")
        labels = ds.labels_dict
        assert len(labels) == 3

    def test_missing_pdbs_raises(self, tmp_path, pdbbind_index):
        records = AffinityFetcher.from_pdbbind_index(str(pdbbind_index))
        with pytest.raises(FileNotFoundError, match="No structure files"):
            AffinityDataset.from_records(records, pdb_dir=tmp_path, fmt="pdb")


# ── DataSplitter ─────────────────────────────────────────────────────

class TestDataSplitter:

    def _make_dataset(self, n: int = 100):
        """Simple list-based dataset for split testing."""
        return list(range(n))

    def test_random_split_sizes(self):
        ds = self._make_dataset(100)
        train, val, test = DataSplitter.random(ds, val_frac=0.1, test_frac=0.1)
        assert len(train) + len(val) + len(test) == 100
        assert len(test) == 10
        assert len(val) == 10

    def test_random_split_no_overlap(self):
        ds = self._make_dataset(50)
        train, val, test = DataSplitter.random(ds)
        train_set = set(train.indices)
        val_set = set(val.indices)
        test_set = set(test.indices)
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_random_split_deterministic(self):
        ds = self._make_dataset(50)
        t1, v1, te1 = DataSplitter.random(ds, seed=42)
        t2, v2, te2 = DataSplitter.random(ds, seed=42)
        assert t1.indices == t2.indices

    def test_by_family(self):
        ds = self._make_dataset(20)
        families = ["kinase"] * 5 + ["protease"] * 5 + ["gpcr"] * 5 + ["ion_channel"] * 5
        train, val, test = DataSplitter.by_family(
            ds, families, val_frac=0.25, test_frac=0.25,
        )
        assert len(train) + len(val) + len(test) == 20

        # Each family should be entirely in one split
        for split_subset in (train, val, test):
            fams_in_split = {families[i] for i in split_subset.indices}
            for fam in fams_in_split:
                all_idx_of_fam = {i for i, f in enumerate(families) if f == fam}
                assert all_idx_of_fam.issubset(set(split_subset.indices))

    def test_temporal(self):
        ds = self._make_dataset(6)
        years = [2017, 2018, 2019, 2019, 2020, 2021]
        train, val, test = DataSplitter.temporal(
            ds, years, val_cutoff=2019, test_cutoff=2020,
        )
        assert set(train.indices) == {0, 1}
        assert set(val.indices) == {2, 3}
        assert set(test.indices) == {4, 5}
