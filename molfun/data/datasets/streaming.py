"""
Streaming dataset for large-scale protein data from object stores.

Reads data lazily from S3/MinIO/GCS/local without downloading the
full dataset first. Supports shuffle buffering and prefetch.

Usage::

    from molfun.data.datasets import StreamingStructureDataset

    # From S3
    ds = StreamingStructureDataset(
        index_path="s3://my-bucket/pdbbind/index.csv",
        structures_prefix="s3://my-bucket/pdbbind/structures/",
        shuffle_buffer=1000,
    )

    # From MinIO
    ds = StreamingStructureDataset(
        index_path="s3://data/index.csv",
        structures_prefix="s3://data/structures/",
        storage_options={"endpoint_url": "http://localhost:9000"},
    )

    # Works with standard DataLoader
    loader = DataLoader(ds, batch_size=4, num_workers=2)
"""

from __future__ import annotations
from typing import Optional, Callable
import csv
import io
import pickle
import random

import torch
from torch.utils.data import IterableDataset, get_worker_info

from molfun.data.storage import open_path, is_remote
from molfun.data.parsers import PDBParser as _PDBStructureParser
from molfun.data.parsers.base import ParsedStructure


class StreamingStructureDataset(IterableDataset):
    """
    IterableDataset that streams protein structures from any fsspec filesystem.

    Reads an index CSV (pdb_id, affinity, ...) and lazily loads structure
    files (.pkl or .cif) on demand. No full download needed.

    Multi-worker safe: each worker processes a disjoint shard of the index.
    """

    def __init__(
        self,
        index_path: str,
        structures_prefix: str,
        pdb_col: str = "pdb_id",
        label_col: str = "affinity",
        fmt: str = "pkl",
        shuffle_buffer: int = 0,
        max_seq_len: int = 512,
        transform: Optional[Callable] = None,
        storage_options: Optional[dict] = None,
    ):
        """
        Args:
            index_path: Path to CSV index file (local or remote).
            structures_prefix: Directory/prefix containing structure files.
            pdb_col: Column name for PDB ID in the CSV.
            label_col: Column name for the label (affinity).
            fmt: File format for structures: "pkl" (pre-computed features)
                 or "cif"/"pdb" (raw structures, requires BioPython).
            shuffle_buffer: If > 0, maintains a buffer of this size and
                 yields samples randomly from it (reservoir sampling).
            max_seq_len: Crop sequences longer than this.
            transform: Optional transform on feature dicts.
            storage_options: fsspec options (e.g. endpoint_url for MinIO).
        """
        super().__init__()
        self.index_path = index_path
        self.structures_prefix = structures_prefix.rstrip("/")
        self.pdb_col = pdb_col
        self.label_col = label_col
        self.fmt = fmt
        self.shuffle_buffer = shuffle_buffer
        self.max_seq_len = max_seq_len
        self.transform = transform
        self.storage_options = storage_options

        self._index = self._load_index()

    def _load_index(self) -> list[dict]:
        """Load the CSV index into memory (just IDs + labels, small)."""
        rows = []
        with open_path(self.index_path, "r", self.storage_options) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "pdb_id": row[self.pdb_col].strip().lower(),
                    "label": float(row.get(self.label_col, "nan")),
                })
        return rows

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self):
        worker_info = get_worker_info()
        entries = self._index

        if worker_info is not None:
            n = len(entries)
            per_worker = n // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else n
            entries = entries[start:end]

        if self.shuffle_buffer > 0:
            yield from self._buffered_shuffle(entries)
        else:
            yield from self._iterate(entries)

    def _iterate(self, entries):
        for entry in entries:
            sample = self._load_sample(entry)
            if sample is not None:
                yield sample

    def _buffered_shuffle(self, entries):
        """Reservoir-based shuffle buffer for approximate shuffling."""
        buf = []
        it = self._iterate(entries)

        for sample in it:
            buf.append(sample)
            if len(buf) >= self.shuffle_buffer:
                break

        for sample in it:
            idx = random.randint(0, len(buf) - 1)
            yield buf[idx]
            buf[idx] = sample

        random.shuffle(buf)
        yield from buf

    def _load_sample(self, entry: dict) -> Optional[tuple[dict, torch.Tensor]]:
        """Load a single structure + label from the object store."""
        pdb_id = entry["pdb_id"]
        label = torch.tensor([entry["label"]], dtype=torch.float32)
        file_path = f"{self.structures_prefix}/{pdb_id}.{self.fmt}"

        try:
            if self.fmt == "pkl":
                features = self._load_pkl(file_path)
            else:
                features = self._load_structure(file_path)
        except Exception:
            return None

        if self.transform is not None:
            features = self.transform(features)

        return features, label

    def _load_pkl(self, path: str) -> dict:
        with open_path(path, "rb", self.storage_options) as f:
            return pickle.load(f)

    def _load_structure(self, path: str) -> dict:
        """Parse a PDB/CIF file streamed from remote storage."""
        with open_path(path, "r", self.storage_options) as f:
            content = f.read()

        if path.endswith(".cif"):
            try:
                from molfun.data.parsers import CIFParser
                parser = CIFParser(max_seq_len=self.max_seq_len)
            except ImportError:
                raise ImportError("BioPython required for streaming CIF: pip install biopython")
            return parser.parse_text(content).to_dict()
        else:
            parser = _PDBStructureParser(max_seq_len=self.max_seq_len)
            return parser.parse_text(content).to_dict()
