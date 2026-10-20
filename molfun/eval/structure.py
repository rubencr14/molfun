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


def _superpose(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Kabsch: move ``p`` onto ``r`` (optimal rotation + translation)."""
    p_mean, r_mean = p.mean(0), r.mean(0)
    pc, rc = p - p_mean, r - r_mean
    u, _, vt = np.linalg.svd(pc.T @ rc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return pc @ rotation.T + r_mean


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


def tm_score(p: np.ndarray, r: np.ndarray) -> float:
    """TM-score (TM-align), normalised by the reference length."""
    from tmtools import tm_align

    result = tm_align(
        np.ascontiguousarray(p, float), np.ascontiguousarray(r, float), "A" * len(p), "A" * len(r)
    )
    return float(result.tm_norm_chain2)
