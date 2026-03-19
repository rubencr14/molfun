"""
PropertyDataset — unified data wrapper for supervised protein learning.

Lightweight (no torch required) dataset that provides sequences, targets,
and optional PDB paths from multiple sources. Serves as the single data
object that all pipeline steps consume.

Three data typologies, one interface:

1. **Structure fine-tuning** (FAPE): use StructureDataset directly —
   targets are 3D coordinates inside the PDB, no external labels.

2. **Property head** (backbone embeddings + head):
   PropertyDataset provides pdb_paths + targets.
   Embeddings are extracted in a separate step.

3. **Classical ML** (sequence features + sklearn):
   PropertyDataset provides sequences + targets.
   Features are extracted by ProteinFeaturizer.

Usage::

    # From CSV
    ds = PropertyDataset.from_csv("data/affinity.csv",
                                  sequence_col="sequence",
                                  target_col="kd_nm")
    ds.sequences   # ["ACDEF...", ...]
    ds.targets     # np.array([1.2, 3.4, ...])

    # From CSV + PDB directory (for head training)
    ds = PropertyDataset.from_csv("data/labels.csv",
                                  pdb_id_col="pdb_id",
                                  target_col="delta_g",
                                  pdb_dir="data/pdbs/")
    ds.pdb_paths   # [Path("data/pdbs/1abc.cif"), ...]

    # From existing data
    ds = PropertyDataset(sequences=seqs, targets=affinities)

    # From PDB fetch results + CSV labels
    ds = PropertyDataset.from_fetch_and_csv(
        pdb_paths=fetched_paths,
        csv_path="data/labels.csv",
    )
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import csv as _csv
import io
import numpy as np


@dataclass
class PropertyDataset:
    """
    Unified data container for supervised protein property prediction.

    Attributes:
        sequences:  Protein sequences (str). Required for classical ML.
        targets:    Target values (np.ndarray). Required for supervised learning.
        pdb_ids:    PDB identifiers (optional, for linking structures to labels).
        pdb_paths:  Paths to PDB/CIF files (optional, for head training).
        labels:     Human-readable target column name.
        metadata:   Extra per-sample metadata dicts.
    """

    sequences: list[str] = field(default_factory=list)
    targets: Optional[np.ndarray] = None
    pdb_ids: list[str] = field(default_factory=list)
    pdb_paths: list[Path] = field(default_factory=list)
    target_name: str = "target"
    metadata: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        if self.targets is not None:
            return len(self.targets)
        if self.sequences:
            return len(self.sequences)
        if self.pdb_paths:
            return len(self.pdb_paths)
        return 0

    def __repr__(self) -> str:
        parts = [f"PropertyDataset(n={len(self)}"]
        if self.sequences:
            parts.append(f"sequences={len(self.sequences)}")
        if self.targets is not None:
            parts.append(f"targets={len(self.targets)}")
        if self.pdb_paths:
            parts.append(f"pdb_paths={len(self.pdb_paths)}")
        if self.target_name != "target":
            parts.append(f"target='{self.target_name}'")
        return ", ".join(parts) + ")"

    @property
    def has_sequences(self) -> bool:
        return bool(self.sequences)

    @property
    def has_targets(self) -> bool:
        return self.targets is not None and len(self.targets) > 0

    @property
    def has_structures(self) -> bool:
        return bool(self.pdb_paths)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        sequence_col: str = "sequence",
        target_col: str = "target",
        pdb_id_col: str = "pdb_id",
        sep: Optional[str] = None,
        pdb_dir: Optional[str | Path] = None,
        pdb_fmt: str = "cif",
    ) -> PropertyDataset:
        """
        Load dataset from a CSV/TSV file.

        Flexible: any combination of columns is supported. Missing columns
        are simply left empty.

        Args:
            csv_path: Path to CSV/TSV file.
            sequence_col: Column name for sequences.
            target_col: Column name for target values.
            pdb_id_col: Column name for PDB identifiers.
            sep: Separator (auto-detected if None).
            pdb_dir: If given, resolves PDB paths from pdb_id column.
            pdb_fmt: PDB file extension ("cif" or "pdb").
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        text = path.read_text()
        if sep is None:
            sep = "\t" if "\t" in text.split("\n")[0] else ","

        reader = _csv.DictReader(io.StringIO(text), delimiter=sep)
        rows = list(reader)
        if not rows:
            raise ValueError(f"Empty CSV file: {csv_path}")

        cols = set(rows[0].keys())

        sequences = []
        targets_list = []
        pdb_ids = []
        pdb_paths = []
        metadata = []

        has_seq = sequence_col in cols
        has_target = target_col in cols
        has_pdb = pdb_id_col in cols
        pdb_dir_path = Path(pdb_dir) if pdb_dir else None

        for row in rows:
            if has_seq:
                sequences.append(row[sequence_col])
            if has_target:
                targets_list.append(float(row[target_col]))
            if has_pdb:
                pid = row[pdb_id_col].strip()
                pdb_ids.append(pid)
                if pdb_dir_path:
                    pdb_paths.append(pdb_dir_path / f"{pid}.{pdb_fmt}")

            extra = {k: v for k, v in row.items()
                     if k not in (sequence_col, target_col, pdb_id_col)}
            if extra:
                metadata.append(extra)

        return cls(
            sequences=sequences,
            targets=np.array(targets_list, dtype=np.float64) if targets_list else None,
            pdb_ids=pdb_ids,
            pdb_paths=pdb_paths,
            target_name=target_col if has_target else "target",
            metadata=metadata,
        )

    @classmethod
    def from_fetch_and_csv(
        cls,
        pdb_paths: list[str | Path],
        csv_path: str | Path,
        target_col: str = "target",
        pdb_id_col: str = "pdb_id",
        sep: Optional[str] = None,
    ) -> PropertyDataset:
        """
        Combine PDB paths from a fetch step with labels from a CSV.

        Matches PDB file stems to the pdb_id column. Only structures
        that have a matching label are kept.
        """
        csv_ds = cls.from_csv(csv_path, target_col=target_col,
                              pdb_id_col=pdb_id_col, sep=sep)
        if not csv_ds.has_targets:
            raise ValueError(f"CSV has no '{target_col}' column")

        label_map = {pid.upper(): t for pid, t in zip(csv_ds.pdb_ids, csv_ds.targets)}

        matched_paths = []
        matched_ids = []
        matched_targets = []
        for p in pdb_paths:
            p = Path(p)
            pid = p.stem.split(".")[0].upper()
            if pid in label_map:
                matched_paths.append(p)
                matched_ids.append(pid)
                matched_targets.append(label_map[pid])

        return cls(
            pdb_paths=matched_paths,
            pdb_ids=matched_ids,
            targets=np.array(matched_targets, dtype=np.float64) if matched_targets else None,
            target_name=target_col,
        )

    @classmethod
    def from_inline(
        cls,
        sequences: Optional[list[str]] = None,
        targets: Optional[list[float]] = None,
        pdb_paths: Optional[list[str | Path]] = None,
        target_name: str = "target",
    ) -> PropertyDataset:
        """Create dataset from in-memory data."""
        return cls(
            sequences=sequences or [],
            targets=np.array(targets, dtype=np.float64) if targets else None,
            pdb_paths=[Path(p) for p in pdb_paths] if pdb_paths else [],
            target_name=target_name,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def extract_sequences(self) -> list[str]:
        """
        Extract sequences from PDB files if not already present.

        Requires pdb_paths to be set and molfun parsers to be available.
        """
        if self.sequences:
            return self.sequences

        if not self.pdb_paths:
            raise ValueError("No sequences or pdb_paths to extract from")

        from molfun.data.parsers import auto_parser
        sequences = []
        for p in self.pdb_paths:
            parser = auto_parser(str(p))
            parsed = parser.parse_file(str(p))
            sequences.append(parsed.sequence)
        self.sequences = sequences
        return sequences

    def split(
        self,
        test_frac: float = 0.2,
        seed: int = 42,
    ) -> tuple[PropertyDataset, PropertyDataset]:
        """
        Random train/test split. Returns two new PropertyDataset instances.
        """
        rng = np.random.RandomState(seed)
        n = len(self)
        indices = rng.permutation(n)
        split_idx = int(n * (1 - test_frac))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        return self._subset(train_idx), self._subset(test_idx)

    def _subset(self, indices) -> PropertyDataset:
        return PropertyDataset(
            sequences=[self.sequences[i] for i in indices] if self.sequences else [],
            targets=self.targets[indices] if self.targets is not None else None,
            pdb_ids=[self.pdb_ids[i] for i in indices] if self.pdb_ids else [],
            pdb_paths=[self.pdb_paths[i] for i in indices] if self.pdb_paths else [],
            target_name=self.target_name,
            metadata=[self.metadata[i] for i in indices] if self.metadata else [],
        )

    def to_structure_dataset(self, **kwargs):
        """
        Convert to a torch StructureDataset for DL fine-tuning.

        Requires torch and pdb_paths.
        """
        from molfun.data.datasets.structure import StructureDataset

        if not self.pdb_paths:
            raise ValueError("Cannot convert to StructureDataset without pdb_paths")

        labels = {}
        if self.has_targets and self.pdb_ids:
            labels = dict(zip(self.pdb_ids, self.targets))

        return StructureDataset(
            pdb_paths=[str(p) for p in self.pdb_paths],
            labels=labels,
            **kwargs,
        )

    def to_affinity_dataset(self, **kwargs):
        """
        Convert to a torch AffinityDataset for DL affinity prediction.

        Requires torch, pdb_paths, and a pdb_dir.
        """
        from molfun.data.datasets.affinity import AffinityDataset
        from molfun.data.sources.affinity import AffinityRecord

        if not self.pdb_ids or not self.has_targets:
            raise ValueError("Need pdb_ids and targets for AffinityDataset")

        records = [
            AffinityRecord(pdb_id=pid, affinity=float(t))
            for pid, t in zip(self.pdb_ids, self.targets)
        ]
        pdb_dir = self.pdb_paths[0].parent if self.pdb_paths else Path(".")
        return AffinityDataset.from_records(records, pdb_dir=pdb_dir, **kwargs)

    def to_dict(self) -> dict:
        """Export as a dict suitable for pipeline state."""
        d: dict = {}
        if self.sequences:
            d["sequences"] = self.sequences
        if self.targets is not None:
            d["y"] = self.targets
        if self.pdb_ids:
            d["pdb_ids"] = self.pdb_ids
        if self.pdb_paths:
            d["pdb_paths"] = [str(p) for p in self.pdb_paths]
        d["target_name"] = self.target_name
        return d
