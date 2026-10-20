"""Structure accuracy on CA coordinates: RMSD, lDDT-Cα, TM-score.

Predicted residue *i* corresponds to reference ``label_seq_id == i + 1``;
reference residues without coordinates are skipped (never zero-filled).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from molfun.core.types import SampledStructure

_LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def paired_ca(
    pred: SampledStructure, ref_cif: str | Path, chain: str
) -> tuple[np.ndarray, np.ndarray]:
    """CA pairs ``(pred, ref)`` over residues resolved in both, each ``(n, 3)``."""
    import gemmi

    pred_ca = pred.ca_coords()
    reference = gemmi.read_structure(str(ref_cif))
    pairs_p, pairs_r = [], []
    for residue in reference[0][chain]:
        if not residue.label_seq:
            continue
        index = residue.label_seq - 1
        ca = residue.find_atom("CA", "*")
        if ca is None or not 0 <= index < len(pred_ca) or np.isnan(pred_ca[index]).any():
            continue
        pairs_p.append(pred_ca[index])
        pairs_r.append((ca.pos.x, ca.pos.y, ca.pos.z))
    return np.asarray(pairs_p, float).reshape(-1, 3), np.asarray(pairs_r, float).reshape(-1, 3)


def _kabsch(p: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation that best map ``p`` onto ``r``: ``p @ R.T + t``."""
    p_mean, r_mean = p.mean(0), r.mean(0)
    u, _, vt = np.linalg.svd((p - p_mean).T @ (r - r_mean))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, r_mean - p_mean @ rotation.T


def _superpose(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Kabsch: move ``p`` onto ``r`` (optimal rotation + translation)."""
    rotation, translation = _kabsch(p, r)
    return p @ rotation.T + translation


def ca_rmsd(p: np.ndarray, r: np.ndarray) -> float:
    """CA RMSD (Å) after optimal superposition."""
    return float(np.sqrt(((_superpose(p, r) - r) ** 2).sum(1).mean()))


def lddt_ca(p: np.ndarray, r: np.ndarray, cutoff: float = 15.0) -> float:
    """Global lDDT-Cα: fraction of reference CA–CA distances under ``cutoff``
    preserved within 0.5/1/2/4 Å, averaged over the four thresholds."""
    dr = np.linalg.norm(r[:, None] - r[None], axis=-1)
    dp = np.linalg.norm(p[:, None] - p[None], axis=-1)
    mask = (dr < cutoff) & ~np.eye(len(r), dtype=bool)
    if not mask.any():
        return float("nan")
    diff = np.abs(dp - dr)[mask]
    return float(np.mean([(diff < t).mean() for t in _LDDT_THRESHOLDS]))


def tm_score(p: np.ndarray, r: np.ndarray, max_iter: int = 20) -> float:
    """TM-score over the given residue pairing (``p[i]`` ↔ ``r[i]``),
    normalised by the number of reference residues.

    Unlike TM-align, the pairing is never changed, so a register-shifted
    prediction scores low. The superposition maximising the score is searched
    as the TM-score program does: Kabsch on seed fragments of length L, L/2,
    L/4, … (≥ 4), each refined by re-fitting on the pairs closer than d0_search.
    """
    n = len(r)
    if n == 0:
        return float("nan")
    d0 = max(0.5, 1.24 * (n - 15) ** (1.0 / 3.0) - 1.8) if n > 21 else 0.5
    d0_search = min(max(d0, 4.5), 8.0)

    def score(idx: np.ndarray) -> tuple[float, np.ndarray]:
        rotation, translation = _kabsch(p[idx], r[idx])
        d = np.linalg.norm(p @ rotation.T + translation - r, axis=1)
        return float(np.sum(1.0 / (1.0 + (d / d0) ** 2)) / n), d

    best = 0.0
    length = n
    while length >= min(4, n):
        for start in range(0, n - length + 1, max(1, length // 2)):
            idx = np.arange(start, start + length)
            for _ in range(max_iter):
                value, d = score(idx)
                best = max(best, value)
                cutoff, chosen = d0_search, np.flatnonzero(d < d0_search)
                while len(chosen) < 3 and cutoff < 2 * d0_search + 20:
                    cutoff += 0.5
                    chosen = np.flatnonzero(d < cutoff)
                if len(chosen) < 3 or np.array_equal(chosen, idx):
                    break
                idx = chosen
        if length < 8:
            break
        length //= 2
    return best
