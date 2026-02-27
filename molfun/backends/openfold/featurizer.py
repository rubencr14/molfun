"""
OpenFold featurizer: PDB/mmCIF → full feature dict for structure fine-tuning.

Produces all input features *and* ground truth fields required by
StructureLossHead / AlphaFoldLoss:
  - aatype, residue_index, seq_length, seq_mask
  - all_atom_positions (37-atom), all_atom_mask
  - backbone_rigid_tensor, backbone_rigid_mask
  - atom14_gt_positions, atom14_gt_exists, atom14_alt_gt_*
  - chi_angles_sin_cos, chi_mask
  - pseudo_beta, pseudo_beta_mask
  - MSA features (single-sequence fallback or from file)
  - Template placeholders

Usage:
    featurizer = OpenFoldFeaturizer(config)
    feat = featurizer.from_pdb("1abc.pdb")                      # single PDB
    feat = featurizer.from_pdb("1abc.pdb", msa_path="1abc.a3m") # with MSA

    # Build a dataset
    from torch.utils.data import Dataset
    class StructureDataset(Dataset):
        def __init__(self, pdb_paths, featurizer):
            self.paths = pdb_paths
            self.f = featurizer
        def __len__(self): return len(self.paths)
        def __getitem__(self, i): return self.f.from_pdb(self.paths[i])
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch


# ======================================================================
# Amino acid tables (mirrors OpenFold's residue_constants)
# ======================================================================

from molfun.constants import THREE_TO_ONE as _THREE_TO_ONE


class OpenFoldFeaturizer:
    """
    Converts PDB/mmCIF files into OpenFold-compatible feature dicts.

    The output dict contains:
    - All MSA/sequence input features (for the encoder)
    - All-atom ground truth (for AlphaFoldLoss)

    Two modes:
    - ``full`` (default): all loss terms enabled. Requires full 37-atom
      coordinates and complete MSA.
    - ``fape_only``: only backbone FAPE. Simpler pipeline, works with CA-only
      structures (other atoms set to zero).

    Args:
        config: OpenFold model_config() object. Used to infer feature shapes.
        max_seq_len: Hard cap on sequence length. Longer proteins are cropped.
        num_msa: Number of MSA sequences to use (incl. query). Default 512.
        num_extra_msa: Extra MSA sequences. Default 1024.
    """

    def __init__(
        self,
        config,
        max_seq_len: int = 256,
        num_msa: int = 512,
        num_extra_msa: int = 1024,
    ):
        self.config = config
        self.max_seq_len = max_seq_len
        self.num_msa = num_msa
        self.num_extra_msa = num_extra_msa

        try:
            from openfold.np import residue_constants as rc
            from openfold.data import data_transforms
        except ImportError:
            raise ImportError(
                "OpenFold is required: "
                "pip install git+https://github.com/aqlaboratory/openfold"
            )
        self._rc = rc
        self._dt = data_transforms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def from_pdb(
        self,
        pdb_path: str | Path,
        msa_path: Optional[str | Path] = None,
        chain_id: str = "A",
    ) -> dict:
        """
        Parse a PDB/mmCIF file and return a full OpenFold feature dict.

        Args:
            pdb_path: Path to .pdb or .cif file.
            msa_path: Optional .a3m / .sto MSA file. If None, single-sequence
                      mode is used (limits MSA-dependent loss terms).
            chain_id: Which chain to extract. Default "A".

        Returns:
            Feature dict with all input + ground truth tensors.
        """
        pdb_path = Path(pdb_path)
        seq, atom_pos, atom_mask = self._parse_pdb(pdb_path, chain_id)

        L = min(len(seq), self.max_seq_len)
        seq = seq[:L]
        atom_pos = atom_pos[:L]
        atom_mask = atom_mask[:L]

        protein = self._build_protein_dict(seq, atom_pos, atom_mask)
        self._apply_gt_transforms(protein)

        msa_feats = self._build_msa_features(seq, msa_path, L)
        protein.update(msa_feats)

        template_feats = self._build_template_placeholders(L)
        protein.update(template_feats)

        protein["seq_length"] = torch.tensor([L], dtype=torch.int64)
        protein["seq_mask"] = torch.ones(L, dtype=torch.float32)

        protein = self._add_recycling_dim(protein)
        return protein

    # ------------------------------------------------------------------
    # PDB parsing
    # ------------------------------------------------------------------

    def _parse_pdb(
        self, path: Path, chain_id: str
    ) -> tuple[str, np.ndarray, np.ndarray]:
        """
        Returns (sequence, atom_pos[L,37,3], atom_mask[L,37]).
        Uses BioPython for .cif; falls back to a minimal PDB reader.
        """
        rc = self._rc
        atom_order = {a: i for i, a in enumerate(rc.atom_types)}  # 37 atoms

        try:
            from Bio.PDB import MMCIFParser, PDBParser
            if path.suffix.lower() == ".cif":
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)
            structure = parser.get_structure("s", str(path))
            return self._extract_biopython(structure, chain_id, atom_order)
        except ImportError:
            return self._extract_minimal_pdb(path, chain_id, atom_order)

    def _extract_biopython(self, structure, chain_id, atom_order):
        rc = self._rc
        model = structure[0]
        try:
            chain = model[chain_id]
        except KeyError:
            chain = next(iter(model))

        seq_chars, positions, masks = [], [], []
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue
            resname = res.get_resname().strip()
            aa = _THREE_TO_ONE.get(resname)
            if aa is None:
                continue
            seq_chars.append(aa)
            pos = np.zeros((37, 3), dtype=np.float32)
            mask = np.zeros(37, dtype=np.float32)
            for atom in res:
                idx = atom_order.get(atom.get_name().strip())
                if idx is not None:
                    pos[idx] = atom.get_vector().get_array()
                    mask[idx] = 1.0
            positions.append(pos)
            masks.append(mask)

        seq = "".join(seq_chars)
        return seq, np.stack(positions), np.stack(masks)

    def _extract_minimal_pdb(self, path: Path, chain_id: str, atom_order: dict):
        """Minimal ATOM-line parser, no BioPython required."""
        residues: dict[tuple, dict] = {}
        order = []

        with open(path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                chain = line[21]
                if chain_id != "" and chain != chain_id:
                    continue
                resname = line[17:20].strip()
                resseq = line[22:26].strip()
                atom_name = line[12:16].strip()
                key = (chain, resseq)
                if key not in residues:
                    aa = _THREE_TO_ONE.get(resname)
                    if aa is None:
                        continue
                    residues[key] = {"aa": aa, "pos": np.zeros((37, 3), np.float32),
                                     "mask": np.zeros(37, np.float32)}
                    order.append(key)
                idx = atom_order.get(atom_name)
                if idx is not None:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    residues[key]["pos"][idx] = [x, y, z]
                    residues[key]["mask"][idx] = 1.0

        seq = "".join(residues[k]["aa"] for k in order)
        positions = np.stack([residues[k]["pos"] for k in order])
        masks = np.stack([residues[k]["mask"] for k in order])
        return seq, positions, masks

    # ------------------------------------------------------------------
    # Protein dict construction
    # ------------------------------------------------------------------

    def _build_protein_dict(
        self,
        seq: str,
        atom_pos: np.ndarray,
        atom_mask: np.ndarray,
    ) -> dict:
        rc = self._rc
        L = len(seq)

        aatype = np.array(
            [rc.restype_order.get(aa, rc.restype_num) for aa in seq],
            dtype=np.int64,
        )

        aatype_t = torch.tensor(aatype, dtype=torch.long)

        import torch.nn.functional as F
        target_feat = F.one_hot(aatype_t.clamp(max=21), num_classes=22).float()

        protein = {
            "aatype": aatype_t,
            "residue_index": torch.arange(L, dtype=torch.long),
            "all_atom_positions": torch.tensor(atom_pos, dtype=torch.float32),
            "all_atom_mask": torch.tensor(atom_mask, dtype=torch.float32),
            "target_feat": target_feat,
        }
        return protein

    def _apply_gt_transforms(self, protein: dict) -> None:
        """Apply OpenFold data transforms to generate GT fields in-place."""
        dt = self._dt

        dt.make_atom14_masks(protein)
        dt.atom37_to_frames(protein)
        dt.get_backbone_frames(protein)
        dt.make_atom14_positions(protein)
        dt.atom37_to_torsion_angles("")(protein)
        dt.get_chi_angles(protein)
        dt.make_pseudo_beta("")(protein)

    # ------------------------------------------------------------------
    # MSA features
    # ------------------------------------------------------------------

    def _build_msa_features(
        self, seq: str, msa_path: Optional[Path], L: int
    ) -> dict:
        """
        Build MSA-related tensors.
        Falls back to single-sequence (query only) if no MSA file given.
        """
        rc = self._rc

        if msa_path is not None:
            msa_seqs = self._parse_a3m(msa_path)
        else:
            msa_seqs = [seq]

        msa_seqs = msa_seqs[: self.num_msa]
        N = len(msa_seqs)

        msa = np.zeros((N, L), dtype=np.int64)
        deletion_matrix = np.zeros((N, L), dtype=np.float32)

        for i, s in enumerate(msa_seqs):
            for j, aa in enumerate(s[:L]):
                msa[i, j] = rc.restype_order.get(aa, rc.restype_num)

        msa_t = torch.tensor(msa, dtype=torch.long)
        del_t = torch.tensor(deletion_matrix, dtype=torch.float32)

        msa_one_hot = torch.zeros(N, L, 23, dtype=torch.float32)
        for i in range(N):
            for j in range(L):
                idx = int(msa_t[i, j])
                if idx < 23:
                    msa_one_hot[i, j, idx] = 1.0

        msa_feat = torch.zeros(N, L, 49, dtype=torch.float32)
        msa_feat[:, :, :23] = msa_one_hot
        has_del = (del_t > 0).float().unsqueeze(-1)
        msa_feat[:, :, 25:26] = has_del

        msa_mask = torch.ones(N, L, dtype=torch.float32)

        extra_N = min(N, self.num_extra_msa)
        extra_msa = msa_t[:extra_N]
        extra_del = del_t[:extra_N]
        extra_mask = msa_mask[:extra_N]
        extra_has_del = has_del[:extra_N]

        return {
            "msa": msa_t,
            "deletion_matrix": del_t,
            "msa_mask": msa_mask,
            "msa_feat": msa_feat,
            "extra_msa": extra_msa,
            "extra_msa_deletion_value": extra_del,
            "extra_msa_mask": extra_mask,
            "extra_has_deletion": extra_has_del.squeeze(-1),
            "extra_deletion_value": extra_del,
            "bert_mask": torch.zeros(N, L, dtype=torch.float32),
            "true_msa": msa_t.clone(),
        }

    @staticmethod
    def _parse_a3m(path) -> list[str]:
        """Parse a .a3m MSA file into a list of (gap-stripped) sequences."""
        seqs = []
        cur = []
        with open(path) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    if cur:
                        seqs.append("".join(cur).replace("-", "").upper())
                    cur = []
                elif line:
                    cur.append(line)
        if cur:
            seqs.append("".join(cur).replace("-", "").upper())
        return seqs

    # ------------------------------------------------------------------
    # Template placeholders
    # ------------------------------------------------------------------

    def _build_template_placeholders(self, L: int) -> dict:
        """Return zero-filled template tensors (no templates)."""
        T = 4  # max_templates
        return {
            "template_aatype": torch.zeros(T, L, dtype=torch.long),
            "template_all_atom_positions": torch.zeros(T, L, 37, 3, dtype=torch.float32),
            "template_all_atom_mask": torch.zeros(T, L, 37, dtype=torch.float32),
            "template_mask": torch.zeros(T, dtype=torch.float32),
            "template_pseudo_beta": torch.zeros(T, L, 3, dtype=torch.float32),
            "template_pseudo_beta_mask": torch.zeros(T, L, dtype=torch.float32),
            "template_torsion_angles_sin_cos": torch.zeros(T, L, 7, 2, dtype=torch.float32),
            "template_alt_torsion_angles_sin_cos": torch.zeros(T, L, 7, 2, dtype=torch.float32),
            "template_torsion_angles_mask": torch.zeros(T, L, 7, dtype=torch.float32),
            "template_sum_probs": torch.zeros(T, 1, dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # Recycling dim
    # ------------------------------------------------------------------

    @staticmethod
    def _add_recycling_dim(protein: dict) -> dict:
        """
        Add a trailing recycling dimension R=1 to every tensor in the dict.

        OpenFold's recycling loop uses ``batch[k][..., cycle_no]`` to extract
        features for each recycling iteration, so ALL tensors must carry this
        trailing dimension (including MSA and template tensors).
        """
        skip = {"seq_length"}
        out = {}
        for k, v in protein.items():
            if k in skip or not isinstance(v, torch.Tensor):
                out[k] = v
            else:
                out[k] = v.unsqueeze(-1)
        return out
