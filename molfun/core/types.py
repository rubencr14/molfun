from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class TrunkOutput:
    """Normalized output from any model adapter.

    Generative backends (e.g. SimpleFold) may leave ``single_repr`` unset and
    report their training losses in ``extra["losses"]`` instead.
    """

    single_repr: torch.Tensor | None = None  # [B, L, D_s]
    pair_repr: torch.Tensor | None = None  # [B, L, L, D_p]
    structure_coords: torch.Tensor | None = None  # [B, L, 3] or [B, L, 14, 3]
    confidence: torch.Tensor | None = None  # [B, L]
    extra: dict = field(default_factory=dict)


@dataclass
class Batch:
    """Minimal batch container for training/inference."""

    sequences: list[str]
    labels: torch.Tensor | None = None
    metadata: dict | None = None


@dataclass
class SampledStructure:
    """One structure sampled by a generative backend.

    Attributes:
        sequence: The input sequence (one letter per residue).
        mmcif: The sampled structure as mmCIF text.
        plddt: Optional per-residue confidence, if the backend predicts it.
    """

    sequence: str
    mmcif: str
    plddt: np.ndarray | None = None

    def ca_coords(self) -> np.ndarray:
        """Return CA coordinates as ``(len(sequence), 3)``, indexed by
        ``label_seq_id - 1`` of the first chain; residues without a CA are NaN."""
        import gemmi

        block = gemmi.cif.read_string(self.mmcif).sole_block()
        structure = gemmi.make_structure_from_block(block)
        coords = np.full((len(self.sequence), 3), np.nan)
        for residue in structure[0][0]:
            index = residue.label_seq - 1 if residue.label_seq else None
            if index is None or not 0 <= index < len(self.sequence):
                continue
            ca = residue.find_atom("CA", "*")
            if ca is not None:
                coords[index] = (ca.pos.x, ca.pos.y, ca.pos.z)
        return coords
