"""
Tests for the data pipeline: sources, datasets, splits, collections.
Uses mock/local data to avoid network calls.
"""

import pytest
import tempfile
import json
import urllib.error
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
from torch.utils.data import DataLoader

from molfun.data.sources.pdb import (
    PDBFetcher,
    StructureRecord,
    deduplicate_by_sequence,
    _pfam_query,
    _ec_query,
    _go_query,
    _taxonomy_query,
    _keyword_query,
    _scop_query,
    _pfam_node,
    _ec_node,
    _go_node,
    _taxonomy_node,
    _keyword_node,
    _resolution_node,
    _and_query,
    _cluster_greedy,
)
from molfun.data.sources.affinity import AffinityFetcher, AffinityRecord
from molfun.data.sources.msa import MSAProvider
from molfun.data.datasets.structure import StructureDataset, collate_structure_batch
from molfun.data.datasets.affinity import AffinityDataset
from molfun.data.splits import DataSplitter
from molfun.data.collections import (
    COLLECTIONS,
    CollectionSpec,
    list_collections,
    fetch_collection,
    count_collection,
    count_all_collections,
)


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


# ── MSAProvider ───────────────────────────────────────────────────────

SAMPLE_A3M = """\
>query
AGVMK
>hit1
AG-MK
>hit2
aAGVmMK
>hit3
XGVMK
"""


class TestMSAProvider:

    def test_single_sequence_backend(self, tmp_path):
        msa = MSAProvider("single", msa_dir=str(tmp_path))
        feats = msa.get("AGVMK", "test1")
        assert feats["msa"].shape == (1, 5)
        assert feats["deletion_matrix"].shape == (1, 5)
        assert feats["msa_mask"].shape == (1, 5)
        assert feats["msa_mask"].sum() == 5

    def test_single_sequence_values(self, tmp_path):
        msa = MSAProvider("single", msa_dir=str(tmp_path))
        feats = msa.get("AG", "test2")
        assert feats["msa"][0, 0].item() == 0   # A
        assert feats["msa"][0, 1].item() == 7   # G

    def test_precomputed_a3m(self, tmp_path):
        (tmp_path / "test3.a3m").write_text(SAMPLE_A3M)
        msa = MSAProvider("precomputed", msa_dir=str(tmp_path))
        feats = msa.get("AGVMK", "test3")
        assert feats["msa"].shape[0] == 4  # query + 3 hits
        assert feats["msa"].shape[1] == 5  # query length
        assert feats["msa_mask"].shape == feats["msa"].shape

    def test_a3m_deletions_counted(self, tmp_path):
        (tmp_path / "test4.a3m").write_text(SAMPLE_A3M)
        msa = MSAProvider("precomputed", msa_dir=str(tmp_path))
        feats = msa.get("AGVMK", "test4")
        # hit2 = "aAGVmMK" → lowercase 'a' before A → del=1 at pos 0
        #                   → lowercase 'm' before M → del=1 at pos 3
        assert feats["deletion_matrix"][2, 0].item() == 1.0
        assert feats["deletion_matrix"][2, 3].item() == 1.0

    def test_a3m_gap_mask(self, tmp_path):
        (tmp_path / "test5.a3m").write_text(SAMPLE_A3M)
        msa = MSAProvider("precomputed", msa_dir=str(tmp_path))
        feats = msa.get("AGVMK", "test5")
        # hit1 = "AG-MK" → gap at position 2, msa_mask should be 0 there
        assert feats["msa_mask"][1, 2].item() == 0.0
        assert feats["msa_mask"][1, 0].item() == 1.0

    def test_cache_saves_and_loads(self, tmp_path):
        msa = MSAProvider("single", msa_dir=str(tmp_path))
        feats1 = msa.get("AGVMK", "cached1")
        # Save happens internally, now load from cache
        cache_file = tmp_path / "cached1.msa.pt"
        assert not cache_file.exists()  # single backend doesn't cache

        # Precomputed does cache after first load
        (tmp_path / "cached2.a3m").write_text(SAMPLE_A3M)
        msa2 = MSAProvider("precomputed", msa_dir=str(tmp_path))
        feats2 = msa2.get("AGVMK", "cached2")
        assert (tmp_path / "cached2.msa.pt").exists()

        # Second call should use cache
        feats3 = msa2.get("AGVMK", "cached2")
        assert torch.equal(feats2["msa"], feats3["msa"])

    def test_precomputed_missing_raises(self, tmp_path):
        msa = MSAProvider("precomputed", msa_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError, match="No A3M file"):
            msa.get("AGVMK", "nonexistent")

    def test_invalid_backend_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown backend"):
            MSAProvider("wrong", msa_dir=str(tmp_path))

    def test_max_msa_depth(self, tmp_path):
        (tmp_path / "deep.a3m").write_text(SAMPLE_A3M)
        msa = MSAProvider("precomputed", msa_dir=str(tmp_path), max_msa_depth=2)
        feats = msa.get("AGVMK", "deep")
        assert feats["msa"].shape[0] == 2  # only query + 1 hit

    def test_gzipped_a3m(self, tmp_path):
        import gzip
        with gzip.open(tmp_path / "gz1.a3m.gz", "wt") as f:
            f.write(SAMPLE_A3M)
        msa = MSAProvider("precomputed", msa_dir=str(tmp_path))
        feats = msa.get("AGVMK", "gz1")
        assert feats["msa"].shape[0] == 4


# ── MSA + StructureDataset integration ───────────────────────────────

class TestMSAIntegration:

    def test_dataset_with_single_msa(self, pdb_dir, tmp_path):
        msa = MSAProvider("single", msa_dir=str(tmp_path))
        paths = [pdb_dir / "1abc.pdb"]
        ds = StructureDataset(pdb_paths=paths, msa_provider=msa)
        feat, _ = ds[0]
        assert "msa" in feat
        assert "deletion_matrix" in feat
        assert "msa_mask" in feat
        assert feat["msa"].shape[1] == len(feat["sequence"])

    def test_dataset_without_msa(self, pdb_dir):
        paths = [pdb_dir / "1abc.pdb"]
        ds = StructureDataset(pdb_paths=paths)
        feat, _ = ds[0]
        assert "msa" not in feat

    def test_collate_with_msa(self, pdb_dir, tmp_path):
        msa = MSAProvider("single", msa_dir=str(tmp_path))
        paths = list(pdb_dir.glob("*.pdb"))
        ds = StructureDataset(
            pdb_paths=paths,
            labels={"1abc": 6.5, "2xyz": 7.2, "3def": 5.8},
            msa_provider=msa,
        )
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_structure_batch)
        batch_feat, batch_labels = next(iter(loader))
        assert "msa" in batch_feat
        assert batch_feat["msa"].shape[0] == 2       # batch dim
        assert batch_feat["msa"].dim() == 3           # [B, N, L]
        assert batch_feat["msa_mask"].shape == batch_feat["msa"].shape


# ── RCSB Query Builders ──────────────────────────────────────────────

class TestQueryBuilders:

    def test_resolution_node_structure(self):
        node = _resolution_node(2.5)
        assert node["type"] == "terminal"
        assert node["parameters"]["value"] == 2.5
        assert node["parameters"]["operator"] == "less_or_equal"

    def test_pfam_node_structure(self):
        node = _pfam_node("PF00069")
        assert node["parameters"]["value"] == "PF00069"
        assert "annotation_id" in node["parameters"]["attribute"]

    def test_ec_node_strips_wildcard(self):
        node = _ec_node("2.7.*")
        assert node["parameters"]["value"] == "2.7"
        node2 = _ec_node("2.7.11.1")
        assert node2["parameters"]["value"] == "2.7.11.1"

    def test_go_node_structure(self):
        node = _go_node("GO:0004672")
        assert node["parameters"]["value"] == "GO:0004672"

    def test_taxonomy_node_converts_to_str(self):
        node = _taxonomy_node(9606)
        assert node["parameters"]["value"] == "9606"

    def test_keyword_node_uses_full_text(self):
        node = _keyword_node("tyrosine kinase")
        assert node["service"] == "full_text"
        assert node["parameters"]["value"] == "tyrosine kinase"

    def test_and_query_wraps_nodes(self):
        q = _and_query(_pfam_node("PF00069"), _resolution_node(3.0))
        assert q["query"]["logical_operator"] == "and"
        assert len(q["query"]["nodes"]) == 2

    def test_pfam_query_has_two_nodes(self):
        q = _pfam_query("PF00069", 3.0)
        assert len(q["query"]["nodes"]) == 2

    def test_ec_query_has_two_nodes(self):
        q = _ec_query("2.7.11", 2.5)
        assert len(q["query"]["nodes"]) == 2

    def test_go_query_has_two_nodes(self):
        q = _go_query("GO:0004672", 3.0)
        assert len(q["query"]["nodes"]) == 2

    def test_taxonomy_query_has_two_nodes(self):
        q = _taxonomy_query(9606, 3.0)
        assert len(q["query"]["nodes"]) == 2

    def test_keyword_query_has_two_nodes(self):
        q = _keyword_query("helicase", 3.0)
        assert len(q["query"]["nodes"]) == 2

    def test_scop_query_has_two_nodes(self):
        q = _scop_query("b.1.1.1", 3.0)
        assert len(q["query"]["nodes"]) == 2


# ── StructureRecord ──────────────────────────────────────────────────

class TestStructureRecord:

    def test_defaults(self):
        r = StructureRecord(pdb_id="1abc")
        assert r.pdb_id == "1abc"
        assert r.resolution is None
        assert r.ec_numbers == []
        assert r.pfam_ids == []
        assert r.has_ligand is False

    def test_full_record(self):
        r = StructureRecord(
            pdb_id="1abc",
            path="/cache/1abc.cif",
            resolution=1.8,
            method="X-RAY DIFFRACTION",
            organism="Homo sapiens",
            organism_id=9606,
            title="Crystal structure of kinase",
            sequence_length=350,
            ec_numbers=["2.7.11.1"],
            pfam_ids=["PF00069"],
            deposition_date="2020-01-15",
            has_ligand=True,
        )
        assert r.resolution == 1.8
        assert r.organism_id == 9606
        assert "2.7.11.1" in r.ec_numbers


# ── PDBFetcher new methods ───────────────────────────────────────────

class TestPDBFetcherDomain:
    """Tests for new domain-specific fetch methods (mocked network)."""

    def _mock_search(self, return_ids):
        def fake_search(query, max_results=500):
            return return_ids
        return fake_search

    def test_fetch_by_ec(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path))
        with patch.object(fetcher, "_search_rcsb", self._mock_search(["1abc"])):
            with patch.object(fetcher, "fetch", return_value=[str(tmp_path / "1abc.cif")]) as mock_fetch:
                paths = fetcher.fetch_by_ec("2.7.11", max_structures=10)
                mock_fetch.assert_called_once_with(["1abc"])
                assert len(paths) == 1

    def test_fetch_by_go(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path))
        with patch.object(fetcher, "_search_rcsb", self._mock_search(["2xyz"])):
            with patch.object(fetcher, "fetch", return_value=[str(tmp_path / "2xyz.cif")]):
                paths = fetcher.fetch_by_go("GO:0004672")
                assert len(paths) == 1

    def test_fetch_by_taxonomy(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path))
        with patch.object(fetcher, "_search_rcsb", self._mock_search(["3def"])):
            with patch.object(fetcher, "fetch", return_value=[str(tmp_path / "3def.cif")]):
                paths = fetcher.fetch_by_taxonomy(9606)
                assert len(paths) == 1

    def test_fetch_by_keyword(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path))
        with patch.object(fetcher, "_search_rcsb", self._mock_search(["4ghi"])):
            with patch.object(fetcher, "fetch", return_value=[str(tmp_path / "4ghi.cif")]):
                paths = fetcher.fetch_by_keyword("tyrosine kinase")
                assert len(paths) == 1

    def test_fetch_by_scop(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path))
        with patch.object(fetcher, "_search_rcsb", self._mock_search(["5jkl"])):
            with patch.object(fetcher, "fetch", return_value=[str(tmp_path / "5jkl.cif")]):
                paths = fetcher.fetch_by_scop("b.1.1.1")
                assert len(paths) == 1

    def test_fetch_combined_requires_filter(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path))
        with pytest.raises(ValueError, match="at least one filter"):
            fetcher.fetch_combined(resolution_max=3.0)

    def test_fetch_combined_builds_query(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path))
        with patch.object(fetcher, "_search_rcsb", self._mock_search(["1abc", "2xyz"])):
            with patch.object(fetcher, "fetch", return_value=["a", "b"]):
                paths = fetcher.fetch_combined(pfam_id="PF00069", taxonomy_id=9606)
                assert len(paths) == 2

    def test_search_ids_returns_ids_without_download(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path))
        with patch.object(fetcher, "_search_rcsb", self._mock_search(["1abc", "2xyz", "3def"])):
            ids = fetcher.search_ids(pfam_id="PF00069", max_results=100)
            assert ids == ["1abc", "2xyz", "3def"]

    def test_search_ids_requires_filter(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path))
        with pytest.raises(ValueError, match="at least one filter"):
            fetcher.search_ids(max_results=10)


# ── Deduplication ────────────────────────────────────────────────────

class TestDeduplication:

    def test_cluster_greedy_identical_sequences(self):
        seqs = {
            "1abc": "AGVMKRPQLF",
            "2xyz": "AGVMKRPQLF",
            "3def": "COMPLETELY_DIFFERENT_SEQ",
        }
        reps = _cluster_greedy(seqs, identity=0.5)
        assert len(reps) == 2
        assert "3def" in reps

    def test_cluster_greedy_all_different(self):
        seqs = {
            "1abc": "AAAAAAAAAA",
            "2xyz": "BBBBBBBBBB",
            "3def": "CCCCCCCCCC",
        }
        reps = _cluster_greedy(seqs, identity=0.5)
        assert len(reps) == 3

    def test_cluster_greedy_all_similar(self):
        seqs = {
            "1abc": "AGVMKRPQLF",
            "2xyz": "AGVMKRPQLA",
            "3def": "AGVMKRPQLG",
        }
        reps = _cluster_greedy(seqs, identity=0.5)
        assert len(reps) == 1

    def test_cluster_greedy_empty(self):
        reps = _cluster_greedy({}, identity=0.3)
        assert reps == []

    def test_cluster_greedy_short_sequences(self):
        seqs = {"1abc": "AG", "2xyz": "AG"}
        reps = _cluster_greedy(seqs, identity=0.5)
        assert len(reps) == 1

    def test_deduplicate_single_id(self):
        reps = deduplicate_by_sequence(["1abc"])
        assert reps == ["1abc"]

    def test_deduplicate_empty(self):
        reps = deduplicate_by_sequence([])
        assert reps == []

    @patch("molfun.data.sources.pdb._fetch_sequences_rcsb", return_value={})
    def test_deduplicate_no_sequences_returns_all(self, mock_fetch):
        reps = deduplicate_by_sequence(["1abc", "2xyz"])
        assert reps == ["1abc", "2xyz"]

    @patch("molfun.data.sources.pdb._mmseqs_available", return_value=False)
    @patch("molfun.data.sources.pdb._fetch_sequences_rcsb")
    def test_deduplicate_uses_greedy_fallback(self, mock_fetch_seq, mock_mmseqs):
        mock_fetch_seq.return_value = {
            "1abc": "AGVMKRPQLF",
            "2xyz": "AGVMKRPQLF",
            "3def": "TOTALLYDIFFERENT",
        }
        reps = deduplicate_by_sequence(["1abc", "2xyz", "3def"], identity=0.5)
        assert len(reps) == 2


# ── Collections ──────────────────────────────────────────────────────

class TestCollections:

    def test_collections_populated(self):
        assert len(COLLECTIONS) > 0
        assert "kinases" in COLLECTIONS
        assert "gpcr" in COLLECTIONS
        assert "sars_cov2" in COLLECTIONS

    def test_collection_spec_fields(self):
        kinases = COLLECTIONS["kinases"]
        assert kinases.pfam_id == "PF00069"
        assert "kinase" in kinases.tags
        assert kinases.description

    def test_kinases_human_has_taxonomy(self):
        kh = COLLECTIONS["kinases_human"]
        assert kh.pfam_id == "PF00069"
        assert kh.taxonomy_id == 9606

    def test_list_collections_all(self):
        specs = list_collections()
        assert len(specs) == len(COLLECTIONS)

    def test_list_collections_by_tag(self):
        kinase_specs = list_collections(tag="kinase")
        assert all("kinase" in s.tags for s in kinase_specs)
        assert len(kinase_specs) >= 2  # at least kinases + kinases_human

    def test_list_collections_empty_tag(self):
        specs = list_collections(tag="nonexistent_tag_xyz")
        assert specs == []

    def test_fetch_collection_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown collection"):
            fetch_collection("does_not_exist")

    @patch("molfun.data.collections.PDBFetcher")
    def test_fetch_collection_kinases(self, mock_cls):
        mock_fetcher = MagicMock()
        mock_cls.return_value = mock_fetcher
        mock_fetcher._search_rcsb.return_value = ["1abc", "2xyz"]
        mock_fetcher.fetch.return_value = ["/path/1abc.cif", "/path/2xyz.cif"]

        paths = fetch_collection("kinases", max_structures=10)
        assert len(paths) == 2
        mock_fetcher.fetch.assert_called_once()

    @patch("molfun.data.collections.PDBFetcher")
    def test_fetch_collection_with_dedup(self, mock_cls):
        mock_fetcher = MagicMock()
        mock_cls.return_value = mock_fetcher
        mock_fetcher._search_rcsb.return_value = ["1abc", "2xyz", "3def"]
        mock_fetcher.fetch.return_value = ["/path/1abc.cif"]

        with patch("molfun.data.sources.pdb.deduplicate_by_sequence", return_value=["1abc"]):
            paths = fetch_collection("kinases", deduplicate=True, identity=0.3)
            assert len(paths) == 1

    @patch("molfun.data.collections.PDBFetcher")
    def test_fetch_collection_combined_query(self, mock_cls):
        mock_fetcher = MagicMock()
        mock_cls.return_value = mock_fetcher
        mock_fetcher.search_ids.return_value = ["1abc"]
        mock_fetcher.fetch.return_value = ["/path/1abc.cif"]

        paths = fetch_collection("kinases_human", max_structures=5)
        assert len(paths) == 1
        mock_fetcher.search_ids.assert_called_once()


# ── Metadata enrichment ──────────────────────────────────────────────

class TestMetadataEnrichment:

    @patch("molfun.data.sources.pdb._fetch_metadata_graphql")
    def test_fetch_with_metadata(self, mock_graphql, tmp_path):
        mock_graphql.return_value = {
            "1ABC": {
                "resolution": 1.8,
                "method": "X-RAY DIFFRACTION",
                "organism": "Homo sapiens",
                "organism_id": 9606,
                "title": "Kinase structure",
                "sequence_length": 300,
                "ec_numbers": ["2.7.11.1"],
                "pfam_ids": ["PF00069"],
                "deposition_date": "2020-01-15",
                "has_ligand": True,
            },
        }
        fetcher = PDBFetcher(cache_dir=str(tmp_path))
        (tmp_path / "1abc.cif").write_text("mock")

        records = fetcher.fetch_with_metadata(["1abc"])
        assert len(records) == 1
        r = records[0]
        assert isinstance(r, StructureRecord)
        assert r.pdb_id == "1abc"
        assert r.resolution == 1.8
        assert r.organism == "Homo sapiens"
        assert "2.7.11.1" in r.ec_numbers
        assert "PF00069" in r.pfam_ids
        assert r.has_ligand is True


# ── Count queries ────────────────────────────────────────────────────

class TestCountQueries:

    def test_count_requires_filter(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path))
        with pytest.raises(ValueError, match="at least one filter"):
            fetcher.count(resolution_max=3.0)

    def test_count_uses_search_rcsb_full(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path))
        with patch.object(fetcher, "_search_rcsb_full", return_value=(1234, [])):
            n = fetcher.count(pfam_id="PF00069")
            assert n == 1234

    def test_count_collection_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown collection"):
            count_collection("does_not_exist")

    @patch("molfun.data.collections.PDBFetcher")
    def test_count_collection_returns_int(self, mock_cls):
        mock_fetcher = MagicMock()
        mock_cls.return_value = mock_fetcher
        mock_fetcher.count.return_value = 5678
        n = count_collection("kinases")
        assert n == 5678
        mock_fetcher.count.assert_called_once()

    @patch("molfun.data.collections.PDBFetcher")
    def test_count_all_collections(self, mock_cls):
        mock_fetcher = MagicMock()
        mock_cls.return_value = mock_fetcher
        mock_fetcher.count.return_value = 100
        counts = count_all_collections()
        assert isinstance(counts, dict)
        assert len(counts) == len(COLLECTIONS)
        assert all(v == 100 for v in counts.values())

    @patch("molfun.data.collections.PDBFetcher")
    def test_count_all_collections_with_tag(self, mock_cls):
        mock_fetcher = MagicMock()
        mock_cls.return_value = mock_fetcher
        mock_fetcher.count.return_value = 42
        counts = count_all_collections(tag="kinase")
        assert len(counts) >= 2
        assert all("kinase" in COLLECTIONS[name].tags for name in counts)


# ── Parallel download & retry ────────────────────────────────────────

class TestParallelDownload:

    def test_fetch_sequential_workers_1(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path), workers=1)
        # Pre-create cached files
        (tmp_path / "1abc.cif").write_text("mock1")
        (tmp_path / "2xyz.cif").write_text("mock2")
        paths = fetcher.fetch(["1abc", "2xyz"])
        assert len(paths) == 2

    def test_fetch_parallel_uses_cache(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path), workers=4)
        (tmp_path / "1abc.cif").write_text("cached")
        (tmp_path / "2xyz.cif").write_text("cached")
        paths = fetcher.fetch(["1abc", "2xyz"])
        assert len(paths) == 2

    def test_fetch_workers_override(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path), workers=1)
        (tmp_path / "a.cif").write_text("ok")
        (tmp_path / "b.cif").write_text("ok")
        paths = fetcher.fetch(["a", "b"], workers=8)
        assert len(paths) == 2

    def test_retry_on_transient_error(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path))
        call_count = {"n": 0}

        def flaky_download(pdb_id, dest):
            call_count["n"] += 1
            if call_count["n"] < 3:
                err = urllib.error.HTTPError(
                    "http://test", 503, "Service Unavailable", {}, None,
                )
                raise err
            Path(dest).write_text("ok")

        with patch.object(fetcher, "_download", side_effect=flaky_download):
            fetcher._download_with_retry("test", str(tmp_path / "test.cif"), max_retries=3)
        assert call_count["n"] == 3

    def test_retry_raises_after_max_attempts(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path))

        def always_fail(pdb_id, dest):
            raise urllib.error.HTTPError(
                "http://test", 503, "Service Unavailable", {}, None,
            )

        with patch.object(fetcher, "_download", side_effect=always_fail):
            with pytest.raises(urllib.error.HTTPError):
                fetcher._download_with_retry("test", str(tmp_path / "t.cif"), max_retries=2)

    def test_retry_no_retry_on_404(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path))
        call_count = {"n": 0}

        def not_found(pdb_id, dest):
            call_count["n"] += 1
            raise FileNotFoundError("not found")

        with patch.object(fetcher, "_download", side_effect=not_found):
            with pytest.raises(FileNotFoundError):
                fetcher._download_with_retry("test", str(tmp_path / "t.cif"), max_retries=3)
        assert call_count["n"] == 1

    def test_default_workers_and_progress(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path), workers=8, progress=True)
        assert fetcher.workers == 8
        assert fetcher.progress is True

    def test_fetch_preserves_order(self, tmp_path):
        fetcher = PDBFetcher(cache_dir=str(tmp_path), workers=4)
        for name in ["c", "a", "b"]:
            (tmp_path / f"{name}.cif").write_text("ok")
        paths = fetcher.fetch(["c", "a", "b"])
        assert paths[0].endswith("c.cif")
        assert paths[1].endswith("a.cif")
        assert paths[2].endswith("b.cif")
