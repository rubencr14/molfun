"""SimpleFold training records: a group-balanced dataset over processed targets.

Records are produced by upstream's ``process_mmcif`` (targets) and
``process_structure`` (tokens). Each dataset item is one *group* (e.g. one
UniProt accession): a random member is drawn per access, so a protein with
hundreds of structures counts once per epoch. Cropping and featurization
reuse upstream's Boltz pipeline with the settings of its training configs.
"""

from __future__ import annotations

import json
import pickle
import random
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from molfun.backends.simplefold.imports import import_simplefold


class SimpleFoldRecordDataset(Dataset):
    """One item per group; each access featurizes a random member record."""

    def __init__(
        self,
        tokenized_dir: str | Path,
        target_dir: str | Path,
        groups: dict[str, list[str]],
        max_tokens: int = 256,
        max_atoms: int = 2304,
    ):
        """
        Args:
            tokenized_dir: ``process_structure`` output (``records/``, ``tokens/``).
            target_dir: ``process_mmcif`` output (``structures/``), used if a token file is missing.
            groups: Group name → record ids.
            max_tokens: Crop size in residues.
            max_atoms: Crop size in atoms.
        """
        if not groups or any(not ids for ids in groups.values()):
            raise ValueError("groups must be non-empty and every group needs at least one record")
        self._sf = import_simplefold()
        self.tokenized_dir = Path(tokenized_dir)
        self.target_dir = Path(target_dir)
        self.group_names = list(groups)
        self.group_members = [list(ids) for ids in groups.values()]
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms
        self._tokenizer = self._sf.BoltzTokenizer()
        self._featurizer = self._sf.BoltzFeaturizer()
        self._cropper = self._sf.BoltzCropper(min_neighborhood=0, max_neighborhood=40)

    def __len__(self) -> int:
        return len(self.group_members)

    def __getitem__(self, index: int) -> dict:
        record_id = random.choice(self.group_members[index])
        return self.featurize(record_id)

    def featurize(self, record_id: str) -> dict:
        """Load, crop and featurize one record (as upstream's training dataset does)."""
        sf = self._sf
        with open(self.tokenized_dir / "records" / f"{record_id.lower()}.json") as f:
            record = sf.Record(**json.load(f))
        token_path = self.tokenized_dir / "tokens" / f"{record.id}.pkl"
        if token_path.is_file():
            with open(token_path, "rb") as f:
                tokenized = pickle.load(f)
        else:
            tokenized = self._tokenizer.tokenize(sf.load_input(record, self.target_dir))

        max_num_tokens = len(tokenized.tokens)
        if max_num_tokens == 0:
            raise ValueError(f"record {record.id} has no tokens")
        tokenized = self._cropper.crop(
            tokenized, max_atoms=self.max_atoms, max_tokens=self.max_tokens, random=_numpy_rng()
        )
        sequence = sf.extract_sequence_from_tokens(tokenized)
        features = self._featurizer.process(
            tokenized,
            max_atoms=None,
            max_tokens=None,
            symmetries={},
            atoms_per_window_queries=32,
            min_dist=2.0,
            max_dist=22.0,
            num_bins=64,
            compute_symmetries=False,
            rotation_augment_ref_pos=False,
            rotation_augment_coords=True,
        )
        features["aa_seq"] = sequence
        features["record"] = asdict(record)
        features["max_num_tokens"] = torch.tensor(max_num_tokens, dtype=torch.long)
        features["cropped_num_tokens"] = torch.tensor(len(tokenized.tokens), dtype=torch.long)
        return features


def collate_records(items: list[dict]) -> dict:
    """Collate featurized records with upstream's collate (use as ``collate_fn``)."""
    return import_simplefold().collate(items)


def _numpy_rng():
    """A legacy-API NumPy RNG (upstream croppers call ``randint``/``choice``),
    seeded from Python's ``random`` so DataLoader worker seeding applies."""
    import numpy as np

    return np.random.RandomState(random.getrandbits(32))
