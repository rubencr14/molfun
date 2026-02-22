"""
Affinity dataset: combines structure features with binding affinity labels.

Wraps StructureDataset with AffinityRecord labels and provides
convenience constructors from PDBbind or CSV sources.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset

from molfun.data.sources.affinity import AffinityRecord
from molfun.data.datasets.structure import StructureDataset, collate_structure_batch


class AffinityDataset(Dataset):
    """
    Dataset for binding affinity prediction.

    Combines protein structures with scalar affinity labels.
    Each item returns (features_dict, label_tensor).

    Usage:
        # From AffinityRecords + PDB directory
        ds = AffinityDataset.from_records(
            records=records,
            pdb_dir="~/.molfun/pdb_cache",
        )

        # From CSV + PDB directory
        ds = AffinityDataset.from_csv(
            csv_path="data/pdbbind.csv",
            pdb_dir="pdbs/",
        )
    """

    def __init__(
        self,
        structure_dataset: StructureDataset,
    ):
        self._ds = structure_dataset

    @classmethod
    def from_records(
        cls,
        records: list[AffinityRecord],
        pdb_dir: str | Path,
        fmt: str = "cif",
        features_dir: Optional[str | Path] = None,
        max_seq_len: int = 512,
        transform: Optional[Callable] = None,
    ) -> AffinityDataset:
        """
        Build dataset from AffinityRecords + a directory of PDB/mmCIF files.

        Args:
            records: List of AffinityRecord from AffinityFetcher.
            pdb_dir: Directory containing structure files named {pdb_id}.{fmt}.
            fmt: File extension ("cif" or "pdb").
            features_dir: Optional pre-computed features directory.
            max_seq_len: Crop sequences longer than this.
            transform: Optional transform on feature dicts.
        """
        pdb_dir = Path(pdb_dir)
        labels = {}
        pdb_paths = []

        for rec in records:
            path = pdb_dir / f"{rec.pdb_id}.{fmt}"
            if not path.exists():
                continue
            pdb_paths.append(path)
            labels[rec.pdb_id] = rec.affinity

        if not pdb_paths:
            raise FileNotFoundError(
                f"No structure files found in {pdb_dir} for the provided records. "
                f"Expected files named like {{pdb_id}}.{fmt}"
            )

        ds = StructureDataset(
            pdb_paths=pdb_paths,
            labels=labels,
            features_dir=features_dir,
            max_seq_len=max_seq_len,
            transform=transform,
        )
        return cls(ds)

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        pdb_dir: str | Path,
        fmt: str = "cif",
        pdb_col: str = "pdb_id",
        affinity_col: str = "affinity",
        features_dir: Optional[str | Path] = None,
        max_seq_len: int = 512,
        transform: Optional[Callable] = None,
    ) -> AffinityDataset:
        """Build dataset directly from a CSV file + PDB directory."""
        from molfun.data.sources.affinity import AffinityFetcher

        records = AffinityFetcher.from_csv(
            str(csv_path), pdb_col=pdb_col, affinity_col=affinity_col,
        )
        return cls.from_records(
            records, pdb_dir, fmt=fmt, features_dir=features_dir,
            max_seq_len=max_seq_len, transform=transform,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> tuple[dict, torch.Tensor]:
        return self._ds[idx]

    @property
    def pdb_ids(self) -> list[str]:
        return self._ds.pdb_ids

    @property
    def sequences(self) -> list[str]:
        return self._ds.sequences

    @property
    def labels_dict(self) -> dict[str, float]:
        return dict(self._ds.labels)

    @staticmethod
    def collate_fn(batch):
        """Use this as DataLoader collate_fn for variable-length structures."""
        return collate_structure_batch(batch)
