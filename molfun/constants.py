"""
Amino acid constants and physicochemical property tables.

Single source of truth for all residue-related data across molfun.
Import from here instead of defining local copies.

Sections:
    1. Residue mappings (THREE_TO_ONE, ONE_TO_THREE, AA_TO_IDX, ...)
    2. Molecular weights
    3. Hydrophobicity (Kyte-Doolittle)
    4. Charge at pH 7
    5. pKa values (side chains + termini)
    6. Aromatic residues
    7. Secondary structure propensities (Chou-Fasman)
    8. Instability index (DIWV) weights
    9. Atom definitions
"""

# ======================================================================
# 1. Residue mappings
# ======================================================================

STANDARD_AA: str = "ACDEFGHIKLMNPQRSTVWY"

THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}

ONE_TO_THREE: dict[str, str] = {
    v: k for k, v in THREE_TO_ONE.items() if len(k) == 3 and v not in ("U", "O")
}

AA_TO_IDX: dict[str, int] = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7,
    "H": 8, "I": 9, "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15,
    "T": 16, "W": 17, "Y": 18, "V": 19, "X": 20, "-": 21,
}

IDX_TO_AA: dict[int, str] = {v: k for k, v in AA_TO_IDX.items()}

STANDARD_RESIDUES: set[str] = {
    k for k in THREE_TO_ONE if k not in ("MSE", "SEC", "PYL")
}

NUM_RESIDUE_TYPES: int = 22  # 20 standard + X (unknown) + gap

# ======================================================================
# 2. Molecular weights (Da, monoisotopic average)
# ======================================================================

MW: dict[str, float] = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G": 75.03, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}

WATER_LOSS: float = 18.015  # Da lost per peptide bond

# ======================================================================
# 3. Hydrophobicity — Kyte-Doolittle scale
# ======================================================================

HYDROPHOBICITY: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# ======================================================================
# 4. Charge at pH 7
# ======================================================================

CHARGE_PH7: dict[str, float] = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "Q": 0, "E": -1, "G": 0, "H": 0.1, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}

# Sparse version (only non-zero residues)
CHARGE_SPARSE: dict[str, float] = {"R": 1, "K": 1, "D": -1, "E": -1, "H": 0.1}

# ======================================================================
# 5. pKa values
# ======================================================================

PK_SIDE: dict[str, float] = {
    "D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.07,
    "H": 6.00, "K": 10.53, "R": 12.48,
}

PK_NH2: float = 9.69   # N-terminus
PK_COOH: float = 2.34  # C-terminus

# Acidic side chains (lose a proton)
PK_ACIDIC: set[str] = {"D", "E", "C", "Y"}

# ======================================================================
# 6. Aromatic residues
# ======================================================================

AROMATIC: set[str] = {"F", "W", "Y", "H"}

# ======================================================================
# 7. Chou-Fasman secondary structure propensities
# ======================================================================

HELIX_PROPENSITY: dict[str, float] = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}

SHEET_PROPENSITY: dict[str, float] = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}

COIL_PROPENSITY: dict[str, float] = {
    "A": 0.66, "R": 0.95, "N": 1.56, "D": 1.46, "C": 1.19,
    "Q": 0.98, "E": 0.74, "G": 1.56, "H": 0.95, "I": 0.47,
    "L": 0.59, "K": 1.01, "M": 0.60, "F": 0.60, "P": 1.52,
    "S": 1.43, "T": 0.96, "W": 0.96, "Y": 1.14, "V": 0.50,
}

# ======================================================================
# 8. Instability index — DIWV dipeptide weights (Guruprasad et al.)
# ======================================================================

DIWV_WEIGHTS: dict[str, float] = {
    "A": -0.02, "C": 0.01, "D": 0.98, "E": -0.01, "F": 0.03,
    "G": 0.74, "H": 0.60, "I": -0.01, "K": -0.01, "L": 0.00,
    "M": -0.04, "N": 0.06, "P": -0.05, "Q": 0.00, "R": -0.01,
    "S": 0.44, "T": 0.13, "V": -0.01, "W": 0.14, "Y": 0.06,
}

# ======================================================================
# 9. Atom definitions
# ======================================================================

BACKBONE_ATOMS: list[str] = ["N", "CA", "C", "O"]

STANDARD_ATOMS_14: list[str] = [
    "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2",
    "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CZ",
]
