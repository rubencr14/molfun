"""Tests for the streaming dataset."""

import tempfile
import pickle
from pathlib import Path
import pytest

import torch
from torch.utils.data import DataLoader

from molfun.data.datasets.streaming import StreamingStructureDataset


def _create_test_data(tmpdir: str, n: int = 10):
    """Create a minimal CSV index + pickle features for testing."""
    index_path = Path(tmpdir) / "index.csv"
    structures_dir = Path(tmpdir) / "structures"
    structures_dir.mkdir()

    with open(index_path, "w") as f:
        f.write("pdb_id,affinity\n")
        for i in range(n):
            pdb_id = f"prot{i:04d}"
            f.write(f"{pdb_id},{5.0 + i * 0.1}\n")

            features = {
                "sequence": "MKFLA" * 4,
                "residue_index": torch.arange(20),
                "all_atom_positions": torch.randn(20, 3),
                "all_atom_mask": torch.ones(20),
                "seq_length": torch.tensor([20]),
            }
            with open(structures_dir / f"{pdb_id}.pkl", "wb") as pf:
                pickle.dump(features, pf)

    return str(index_path), str(structures_dir)


class TestStreamingStructureDataset:
    def test_basic_iteration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path, structures_dir = _create_test_data(tmpdir, n=5)
            ds = StreamingStructureDataset(
                index_path=index_path,
                structures_prefix=structures_dir,
            )
            assert len(ds) == 5
            samples = list(ds)
            assert len(samples) == 5
            features, label = samples[0]
            assert "sequence" in features
            assert label.shape == (1,)

    def test_with_shuffle_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path, structures_dir = _create_test_data(tmpdir, n=20)
            ds = StreamingStructureDataset(
                index_path=index_path,
                structures_prefix=structures_dir,
                shuffle_buffer=5,
            )
            samples = list(ds)
            assert len(samples) == 20

    def test_with_dataloader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path, structures_dir = _create_test_data(tmpdir, n=8)
            ds = StreamingStructureDataset(
                index_path=index_path,
                structures_prefix=structures_dir,
            )
            loader = DataLoader(ds, batch_size=None)
            count = 0
            for features, label in loader:
                count += 1
                assert isinstance(features, dict)
            assert count == 8

    def test_missing_structure_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path, structures_dir = _create_test_data(tmpdir, n=5)
            # Remove one structure file
            (Path(structures_dir) / "prot0002.pkl").unlink()
            ds = StreamingStructureDataset(
                index_path=index_path,
                structures_prefix=structures_dir,
            )
            samples = list(ds)
            assert len(samples) == 4

    def test_transform(self):
        def add_flag(features):
            features["transformed"] = True
            return features

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path, structures_dir = _create_test_data(tmpdir, n=3)
            ds = StreamingStructureDataset(
                index_path=index_path,
                structures_prefix=structures_dir,
                transform=add_flag,
            )
            sample = next(iter(ds))
            assert sample[0]["transformed"] is True
