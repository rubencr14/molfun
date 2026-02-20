"""
Structure dataset: loads PDB/mmCIF files and produces feature dicts
compatible with OpenFold or ESMFold.

Supports:
- Pre-computed feature pickles (fast, recommended for OpenFold)
- On-the-fly parsing from mmCIF/PDB (extracts sequence + coords)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable
import pickle

import torch
from torch.utils.data import Dataset

try:
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


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
        max_seq_len: int = 512,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            pdb_paths: List of paths to PDB/mmCIF files.
            labels: Optional {pdb_id: value} label mapping.
            features_dir: Directory with pre-computed .pkl feature files.
                          If a .pkl exists for a PDB ID, it is loaded instead
                          of parsing the structure file.
            max_seq_len: Crop sequences longer than this.
            transform: Optional transform applied to the feature dict.
        """
        self.paths = [Path(p) for p in pdb_paths]
        self.labels = labels or {}
        self.features_dir = Path(features_dir) if features_dir else None
        self.max_seq_len = max_seq_len
        self.transform = transform

        self._ids = [p.stem.split(".")[0].lower() for p in self.paths]

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
        # Try pre-computed pickle first
        if self.features_dir is not None:
            pkl_path = self.features_dir / f"{pdb_id}.pkl"
            if pkl_path.exists():
                return self._load_pickle(pkl_path)

        return self._parse_structure(self.paths[idx])

    @staticmethod
    def _load_pickle(path: Path) -> dict:
        with open(path, "rb") as f:
            return pickle.load(f)

    def _parse_structure(self, path: Path) -> dict:
        """Parse PDB/mmCIF → minimal feature dict."""
        suffix = path.suffix.lower()

        if suffix == ".cif" and HAS_BIOPYTHON:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("s", str(path))
        elif suffix == ".pdb" and HAS_BIOPYTHON:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("s", str(path))
        elif HAS_BIOPYTHON:
            for P in (MMCIFParser(QUIET=True), PDBParser(QUIET=True)):
                try:
                    structure = P.get_structure("s", str(path))
                    break
                except Exception:
                    continue
            else:
                raise ValueError(f"Cannot parse {path}")
        else:
            return self._parse_structure_fallback(path)

        return self._extract_features_biopython(structure)

    def _extract_features_biopython(self, structure) -> dict:
        """Extract sequence, CA coords, all-atom coords from BioPython structure."""
        model = structure[0]
        residues = []
        for chain in model:
            for res in chain:
                if res.id[0] != " ":
                    continue
                resname = res.get_resname().strip()
                one_letter = _THREE_TO_ONE.get(resname, "X")
                ca = res["CA"].get_vector().get_array() if "CA" in res else None
                all_atoms = []
                for atom in res:
                    all_atoms.append(atom.get_vector().get_array())
                residues.append((one_letter, ca, all_atoms))

        L = min(len(residues), self.max_seq_len)
        residues = residues[:L]

        sequence = "".join(r[0] for r in residues)

        ca_coords = torch.zeros(L, 3)
        ca_mask = torch.zeros(L)
        for i, (_, ca, _) in enumerate(residues):
            if ca is not None:
                ca_coords[i] = torch.tensor(ca)
                ca_mask[i] = 1.0

        return {
            "sequence": sequence,
            "residue_index": torch.arange(L),
            "all_atom_positions": ca_coords,
            "all_atom_mask": ca_mask,
            "seq_length": torch.tensor([L]),
        }

    def _parse_structure_fallback(self, path: Path) -> dict:
        """
        Minimal parser when BioPython is not available.
        Reads ATOM lines from PDB format or raises for CIF.
        """
        if path.suffix.lower() == ".cif":
            raise ImportError(
                "BioPython is required to parse mmCIF files: pip install biopython"
            )

        residues = []
        seen = set()
        with open(path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                resname = line[17:20].strip()
                chain = line[21]
                resseq = line[22:26].strip()
                atom_name = line[12:16].strip()
                key = (chain, resseq)
                if key not in seen:
                    seen.add(key)
                    one_letter = _THREE_TO_ONE.get(resname, "X")
                    residues.append({"aa": one_letter, "ca": None})
                if atom_name == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    residues[-1]["ca"] = [x, y, z]

        L = min(len(residues), self.max_seq_len)
        residues = residues[:L]
        sequence = "".join(r["aa"] for r in residues)

        ca_coords = torch.zeros(L, 3)
        ca_mask = torch.zeros(L)
        for i, r in enumerate(residues):
            if r["ca"] is not None:
                ca_coords[i] = torch.tensor(r["ca"])
                ca_mask[i] = 1.0

        return {
            "sequence": sequence,
            "residue_index": torch.arange(L),
            "all_atom_positions": ca_coords,
            "all_atom_mask": ca_mask,
            "seq_length": torch.tensor([L]),
        }


def collate_structure_batch(batch: list[tuple[dict, torch.Tensor]]) -> tuple[dict, torch.Tensor]:
    """
    Custom collate function that pads variable-length structures.

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

    return {
        "sequences": sequences,
        "all_atom_positions": positions,
        "all_atom_mask": mask,
        "residue_index": residue_index,
        "seq_length": torch.tensor([f["seq_length"].item() for f in features_list]),
    }, labels
