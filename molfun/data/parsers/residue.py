"""
Shared constants for biological residues and atoms.

Single source of truth — all parsers and consumers import from here.
"""

THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}

ONE_TO_THREE: dict[str, str] = {v: k for k, v in THREE_TO_ONE.items() if len(k) == 3}

AA_TO_IDX: dict[str, int] = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7,
    "H": 8, "I": 9, "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15,
    "T": 16, "W": 17, "Y": 18, "V": 19, "X": 20, "-": 21,
}

IDX_TO_AA: dict[int, str] = {v: k for k, v in AA_TO_IDX.items()}

STANDARD_RESIDUES: set[str] = set(THREE_TO_ONE.keys())

BACKBONE_ATOMS: list[str] = ["N", "CA", "C", "O"]

STANDARD_ATOMS_14: list[str] = [
    "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2",
    "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CZ",
]

NUM_RESIDUE_TYPES: int = 22  # 20 standard + X (unknown) + gap
