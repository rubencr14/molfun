"""
Shared constants for biological residues and atoms.

Re-exports from ``molfun.constants`` for backward compatibility.
All new code should import directly from ``molfun.constants``.
"""

from molfun.constants import (
    AA_TO_IDX,
    BACKBONE_ATOMS,
    IDX_TO_AA,
    NUM_RESIDUE_TYPES,
    ONE_TO_THREE,
    STANDARD_ATOMS_14,
    STANDARD_RESIDUES,
    THREE_TO_ONE,
)

__all__ = [
    "THREE_TO_ONE",
    "ONE_TO_THREE",
    "AA_TO_IDX",
    "IDX_TO_AA",
    "STANDARD_RESIDUES",
    "BACKBONE_ATOMS",
    "STANDARD_ATOMS_14",
    "NUM_RESIDUE_TYPES",
]
