"""
Shared constants for biological residues and atoms.

Re-exports from ``molfun.constants`` for backward compatibility.
All new code should import directly from ``molfun.constants``.
"""

from molfun.constants import (
    THREE_TO_ONE,
    ONE_TO_THREE,
    AA_TO_IDX,
    IDX_TO_AA,
    STANDARD_RESIDUES,
    BACKBONE_ATOMS,
    STANDARD_ATOMS_14,
    NUM_RESIDUE_TYPES,
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
