"""
Protein feature extractors for classical ML.

All featurizers follow the sklearn TransformerMixin interface
(fit / transform / fit_transform).  They accept lists of protein
sequences (str) and return numpy arrays.

Usage::

    from molfun.ml.features import ProteinFeaturizer

    feat = ProteinFeaturizer(features=["aa_composition", "physicochemical"])
    X = feat.fit_transform(sequences)   # np.ndarray [N, D]
"""

from __future__ import annotations
from typing import Optional
import math
import numpy as np

from molfun.data.parsers.residue import AA_TO_IDX


# ------------------------------------------------------------------
# Amino-acid property tables (no external deps)
# ------------------------------------------------------------------

_STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

_MW: dict[str, float] = {
    "A": 89.1, "R": 174.2, "N": 132.1, "D": 133.1, "C": 121.2,
    "Q": 146.1, "E": 147.1, "G": 75.0, "H": 155.2, "I": 131.2,
    "L": 131.2, "K": 146.2, "M": 149.2, "F": 165.2, "P": 115.1,
    "S": 105.1, "T": 119.1, "W": 204.2, "Y": 181.2, "V": 117.1,
}

_HYDROPHOBICITY: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

_CHARGE_PH7: dict[str, float] = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "Q": 0, "E": -1, "G": 0, "H": 0.1, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}

_PK_SIDE: dict[str, float] = {
    "D": 3.65, "E": 4.25, "H": 6.00, "C": 8.18,
    "Y": 10.07, "K": 10.53, "R": 12.48,
}

_HELIX_PROPENSITY: dict[str, float] = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}

_SHEET_PROPENSITY: dict[str, float] = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}

_COIL_PROPENSITY: dict[str, float] = {
    "A": 0.66, "R": 0.95, "N": 1.56, "D": 1.46, "C": 1.19,
    "Q": 0.98, "E": 0.74, "G": 1.56, "H": 0.95, "I": 0.47,
    "L": 0.59, "K": 1.01, "M": 0.60, "F": 0.60, "P": 1.52,
    "S": 1.43, "T": 0.96, "W": 0.96, "Y": 1.14, "V": 0.50,
}


# ------------------------------------------------------------------
# Individual feature extractors  (sequence → 1-D array)
# ------------------------------------------------------------------

def _aa_composition(seq: str) -> np.ndarray:
    """Frequency of each of the 20 standard amino acids."""
    counts = np.zeros(20, dtype=np.float64)
    for ch in seq.upper():
        idx = _STANDARD_AA.find(ch)
        if idx >= 0:
            counts[idx] += 1
    total = max(len(seq), 1)
    return counts / total


def _dipeptide_composition(seq: str) -> np.ndarray:
    """Frequency of each of the 400 dipeptide pairs."""
    n_aa = len(_STANDARD_AA)
    counts = np.zeros(n_aa * n_aa, dtype=np.float64)
    s = seq.upper()
    for i in range(len(s) - 1):
        a = _STANDARD_AA.find(s[i])
        b = _STANDARD_AA.find(s[i + 1])
        if a >= 0 and b >= 0:
            counts[a * n_aa + b] += 1
    total = max(len(s) - 1, 1)
    return counts / total


def _length(seq: str) -> np.ndarray:
    return np.array([len(seq)], dtype=np.float64)


def _molecular_weight(seq: str) -> np.ndarray:
    mw = sum(_MW.get(ch, 0) for ch in seq.upper())
    mw -= (len(seq) - 1) * 18.015  # water loss per peptide bond
    return np.array([mw], dtype=np.float64)


def _charge_ph7(seq: str) -> np.ndarray:
    charge = sum(_CHARGE_PH7.get(ch, 0) for ch in seq.upper())
    return np.array([charge], dtype=np.float64)


def _isoelectric_point(seq: str) -> np.ndarray:
    """Estimate pI by bisection on Henderson-Hasselbalch."""
    s = seq.upper()
    n_term_pk = 9.69
    c_term_pk = 2.34

    def _net_charge(ph: float) -> float:
        charge = 1.0 / (1.0 + 10 ** (ph - n_term_pk))  # N-term
        charge -= 1.0 / (1.0 + 10 ** (c_term_pk - ph))  # C-term
        for aa in s:
            pk = _PK_SIDE.get(aa)
            if pk is None:
                continue
            if aa in ("D", "E", "C", "Y"):
                charge -= 1.0 / (1.0 + 10 ** (pk - ph))
            else:
                charge += 1.0 / (1.0 + 10 ** (ph - pk))
        return charge

    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _net_charge(mid) > 0:
            lo = mid
        else:
            hi = mid
    return np.array([(lo + hi) / 2], dtype=np.float64)


def _hydrophobicity_stats(seq: str) -> np.ndarray:
    """Mean, std, and autocorrelation(lag=1) of Kyte-Doolittle hydrophobicity."""
    vals = [_HYDROPHOBICITY.get(ch, 0) for ch in seq.upper()]
    if not vals:
        return np.zeros(3, dtype=np.float64)
    arr = np.array(vals, dtype=np.float64)
    mean = arr.mean()
    std = arr.std()
    if len(arr) > 1 and std > 1e-8:
        centered = arr - mean
        autocorr = np.dot(centered[:-1], centered[1:]) / (len(centered) - 1) / (std ** 2)
    else:
        autocorr = 0.0
    return np.array([mean, std, autocorr], dtype=np.float64)


def _aromaticity(seq: str) -> np.ndarray:
    """Fraction of aromatic residues (F, W, Y)."""
    aromatic = sum(1 for ch in seq.upper() if ch in "FWY")
    return np.array([aromatic / max(len(seq), 1)], dtype=np.float64)


def _ss_propensity(seq: str) -> np.ndarray:
    """Mean Chou-Fasman propensities for helix, sheet, coil."""
    h, s, c = 0.0, 0.0, 0.0
    n = 0
    for ch in seq.upper():
        if ch in _HELIX_PROPENSITY:
            h += _HELIX_PROPENSITY[ch]
            s += _SHEET_PROPENSITY[ch]
            c += _COIL_PROPENSITY[ch]
            n += 1
    n = max(n, 1)
    return np.array([h / n, s / n, c / n], dtype=np.float64)


def _sequence_entropy(seq: str) -> np.ndarray:
    """Shannon entropy of the amino acid distribution."""
    comp = _aa_composition(seq)
    comp = comp[comp > 0]
    entropy = -np.sum(comp * np.log2(comp))
    return np.array([entropy], dtype=np.float64)


def _gravy(seq: str) -> np.ndarray:
    """Grand average of hydropathicity (GRAVY score)."""
    if not seq:
        return np.array([0.0], dtype=np.float64)
    total = sum(_HYDROPHOBICITY.get(ch, 0) for ch in seq.upper())
    return np.array([total / len(seq)], dtype=np.float64)


def _tiny_small_aliphatic_aromatic_polar_charged(seq: str) -> np.ndarray:
    """Grouped amino acid fractions (6 groups)."""
    groups = {
        "tiny": "AGCS",
        "small": "ACDGNPSTV",
        "aliphatic": "AILV",
        "aromatic": "FHW",
        "polar": "DEHKNQRST",
        "charged": "DEKR",
    }
    n = max(len(seq), 1)
    s = seq.upper()
    return np.array(
        [sum(1 for ch in s if ch in grp) / n for grp in groups.values()],
        dtype=np.float64,
    )


# ------------------------------------------------------------------
# Feature registry
# ------------------------------------------------------------------

_FEATURE_REGISTRY: dict[str, tuple] = {
    "aa_composition":   (_aa_composition, 20, "Frequency of 20 standard amino acids"),
    "dipeptide":        (_dipeptide_composition, 400, "Frequency of 400 dipeptide pairs"),
    "length":           (_length, 1, "Sequence length"),
    "molecular_weight": (_molecular_weight, 1, "Estimated molecular weight (Da)"),
    "charge_ph7":       (_charge_ph7, 1, "Net charge at pH 7"),
    "isoelectric_point": (_isoelectric_point, 1, "Estimated isoelectric point"),
    "hydrophobicity":   (_hydrophobicity_stats, 3, "Hydrophobicity mean, std, autocorrelation"),
    "aromaticity":      (_aromaticity, 1, "Fraction of aromatic residues"),
    "ss_propensity":    (_ss_propensity, 3, "Chou-Fasman helix/sheet/coil propensities"),
    "sequence_entropy": (_sequence_entropy, 1, "Shannon entropy of AA distribution"),
    "gravy":            (_gravy, 1, "GRAVY score"),
    "aa_groups":        (_tiny_small_aliphatic_aromatic_polar_charged, 6, "Grouped AA fractions (6 groups)"),
}

AVAILABLE_FEATURES: list[str] = sorted(_FEATURE_REGISTRY.keys())

DEFAULT_FEATURES: list[str] = [
    "aa_composition", "length", "molecular_weight",
    "charge_ph7", "hydrophobicity", "aromaticity",
    "ss_propensity", "sequence_entropy",
]


# ------------------------------------------------------------------
# ProteinFeaturizer (sklearn-compatible)
# ------------------------------------------------------------------

class ProteinFeaturizer:
    """
    Extract numerical features from protein sequences.

    Follows the sklearn TransformerMixin contract (fit / transform).

    Args:
        features: List of feature names to compute.
                  See ``AVAILABLE_FEATURES`` for options.
                  Default: a balanced set of ~30 dimensions.

    Usage::

        feat = ProteinFeaturizer(features=["aa_composition", "length"])
        X = feat.fit_transform(["AGVMK", "MTKIL"])
        # X.shape == (2, 21)
    """

    def __sklearn_tags__(self):
        try:
            from sklearn.utils._tags import Tags
            return Tags()
        except ImportError:
            return {}

    def __init__(self, features: Optional[list[str]] = None):
        self.features = features or list(DEFAULT_FEATURES)
        for f in self.features:
            if f not in _FEATURE_REGISTRY:
                available = ", ".join(AVAILABLE_FEATURES)
                raise ValueError(f"Unknown feature '{f}'. Available: {available}")

    def fit(self, X, y=None):
        """No-op (stateless featurizer). Returns self for pipeline compat."""
        return self

    def transform(self, X) -> np.ndarray:
        """
        Transform sequences to feature matrix.

        Args:
            X: List of protein sequences (str).

        Returns:
            np.ndarray of shape ``(len(X), n_features)``.
        """
        rows = []
        for seq in X:
            parts = []
            for fname in self.features:
                fn, _, _ = _FEATURE_REGISTRY[fname]
                parts.append(fn(seq))
            rows.append(np.concatenate(parts))
        return np.vstack(rows)

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def n_features(self) -> int:
        """Total number of feature dimensions."""
        return sum(_FEATURE_REGISTRY[f][1] for f in self.features)

    @property
    def feature_names(self) -> list[str]:
        """Flat list of feature column names."""
        names = []
        for fname in self.features:
            _, dim, _ = _FEATURE_REGISTRY[fname]
            if dim == 1:
                names.append(fname)
            else:
                names.extend(f"{fname}_{i}" for i in range(dim))
        return names

    def describe(self) -> list[dict]:
        """Return metadata for each active feature."""
        return [
            {"name": f, "dims": _FEATURE_REGISTRY[f][1], "description": _FEATURE_REGISTRY[f][2]}
            for f in self.features
        ]

    def get_params(self, deep=True):
        return {"features": self.features}

    def set_params(self, **params):
        if "features" in params:
            self.features = params["features"]
        return self
