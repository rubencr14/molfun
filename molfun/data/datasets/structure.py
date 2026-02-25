"""
Structure dataset: loads PDB/mmCIF files and produces feature dicts
compatible with OpenFold or ESMFold.

Supports:
- Pre-computed feature pickles (fast, recommended for OpenFold)
- On-the-fly parsing from mmCIF/PDB via molfun.data.parsers
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable
import pickle

import torch
from torch.utils.data import Dataset

from molfun.data.parsers import auto_parser, PDBParser
from molfun.data.parsers.residue import THREE_TO_ONE


class StructureDataset(Dataset):
    """
    Dataset of protein structures for fine-tuning.

    Each item returns a dict of features + optional label, ready for
    the model adapter's forward() method.

    Usage:
        # From PDB files (on-the-fly parsing)
        ds = StructureDataset(pdb_paths=["1abc.cif", "2xyz.cif"])

        # From pre-computed feature pickles (OpenFold-style)
        ds = StructureDataset(
            pdb_paths=["1abc.cif"],
            features_dir="features/",   # contains 1abc.pkl
        )

        # With labels
        ds = StructureDataset(
            pdb_paths=["1abc.cif"],
            labels={"1abc": 6.5},
        )
    """

    def __init__(
        self,
        pdb_paths: list[str | Path],
        labels: Optional[dict[str, float]] = None,
        features_dir: Optional[str | Path] = None,
        msa_provider=None,
        max_seq_len: int = 512,
        transform: Optional[Callable] = None,
    ):
        self.paths = [Path(p) for p in pdb_paths]
        self.labels = labels or {}
        self.features_dir = Path(features_dir) if features_dir else None
        self.msa_provider = msa_provider
        self.max_seq_len = max_seq_len
        self.transform = transform

        self._ids = [p.stem.split(".")[0].lower() for p in self.paths]
        self._parser_cache: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[dict, torch.Tensor]:
        pdb_id = self._ids[idx]

        features = self._load_features(idx, pdb_id)

        label = self.labels.get(pdb_id, float("nan"))
        label_t = torch.tensor([label], dtype=torch.float32)

        if self.transform is not None:
            features = self.transform(features)

        return features, label_t

    @property
    def pdb_ids(self) -> list[str]:
        return list(self._ids)

    @property
    def sequences(self) -> list[str]:
        """Extract sequences from all structures (lazy, caches on first call)."""
        if not hasattr(self, "_sequences_cache"):
            self._sequences_cache = []
            for idx in range(len(self)):
                feats = self._load_features(idx, self._ids[idx])
                self._sequences_cache.append(feats.get("sequence", ""))
        return self._sequences_cache

    # ------------------------------------------------------------------
    # Feature loading
    # ------------------------------------------------------------------

    def _load_features(self, idx: int, pdb_id: str) -> dict:
        if self.features_dir is not None:
            pkl_path = self.features_dir / f"{pdb_id}.pkl"
            if pkl_path.exists():
                return self._load_pickle(pkl_path)

        features = self._parse_structure(self.paths[idx])

        if "msa" not in features and self.msa_provider is not None:
            msa_feats = self.msa_provider.get(features["sequence"], pdb_id)
            features.update(msa_feats)

        return features

    @staticmethod
    def _load_pickle(path: Path) -> dict:
        with open(path, "rb") as f:
            return pickle.load(f)

    def _parse_structure(self, path: Path) -> dict:
        """Parse PDB/mmCIF → feature dict using the parser registry."""
        ext = path.suffix.lower()
        if ext not in self._parser_cache:
            try:
                self._parser_cache[ext] = auto_parser(
                    str(path), max_seq_len=self.max_seq_len,
                )
            except ValueError:
                self._parser_cache[ext] = PDBParser(max_seq_len=self.max_seq_len)

        parser = self._parser_cache[ext]
        parsed = parser.parse_file(str(path))
        return parsed.to_dict()


def collate_structure_batch(batch: list[tuple[dict, torch.Tensor]]) -> tuple[dict, torch.Tensor]:
    """
    Custom collate function that pads variable-length structures.

    Handles optional MSA features if present.

    Returns:
        (features_dict, labels) where features_dict has batched tensors.
    """
    features_list, labels = zip(*batch)
    labels = torch.stack(labels)

    max_len = max(f["seq_length"].item() for f in features_list)
    B = len(features_list)

    positions = torch.zeros(B, max_len, 3)
    mask = torch.zeros(B, max_len)
    residue_index = torch.zeros(B, max_len, dtype=torch.long)
    sequences = []

    for i, feat in enumerate(features_list):
        L = feat["seq_length"].item()
        positions[i, :L] = feat["all_atom_positions"]
        mask[i, :L] = feat["all_atom_mask"]
        residue_index[i, :L] = feat["residue_index"]
        sequences.append(feat["sequence"])

    result = {
        "sequences": sequences,
        "all_atom_positions": positions,
        "all_atom_mask": mask,
        "residue_index": residue_index,
        "seq_length": torch.tensor([f["seq_length"].item() for f in features_list]),
    }

    if "msa" in features_list[0]:
        max_msa_depth = max(f["msa"].shape[0] for f in features_list)
        msa_batch = torch.zeros(B, max_msa_depth, max_len, dtype=torch.long)
        del_batch = torch.zeros(B, max_msa_depth, max_len)
        msa_mask_batch = torch.zeros(B, max_msa_depth, max_len)

        for i, feat in enumerate(features_list):
            N, L = feat["msa"].shape
            msa_batch[i, :N, :L] = feat["msa"]
            del_batch[i, :N, :L] = feat["deletion_matrix"]
            msa_mask_batch[i, :N, :L] = feat["msa_mask"]

        result["msa"] = msa_batch
        result["deletion_matrix"] = del_batch
        result["msa_mask"] = msa_mask_batch

    return result, labels
