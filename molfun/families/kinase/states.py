"""Kinase conformational state (DFG in/out × αC in/out) from coordinates.

The motif residues — β3-Lys, αC-Glu and the DFG Phe (KLIFS pocket positions
17, 24 and 82) — are located by aligning the 85-residue KLIFS pocket to the
chain sequence. ``locate_motif`` is ported from
``nomosis-kinases/src/model/experts/dfg_expert.py`` (same aligner settings).

``StateLabeler`` learns, from experimental structures with published KLIFS
labels, two depth-2 decision trees over inter-residue distances, and is
stored as plain JSON. The same labeler is applied to experimental and
predicted structures.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

POS_LYS, POS_GLU, POS_PHE = 16, 23, 81  # KLIFS 17, 24, 82 (0-based)
DFG_FEATURES = ["phe_glu_ca", "phe_lys_ca", "phe_cz_glu_ca", "phe_cz_lys_ca"]
AC_FEATURES = ["lys_glu_ca", "lys_nz_glu_cd"]


def _aligner():
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.match_score = 2
    aligner.mismatch_score = -1
    return aligner


def locate_motif(pocket: str | None, seq: str) -> tuple[int, int, int] | None:
    """0-based indices in ``seq`` of (β3-Lys, αC-Glu, DFG-Phe), or None."""
    if not isinstance(pocket, str) or len(pocket) != 85:
        return None
    alignment = _aligner().align(seq, pocket)[0]
    pocket_to_seq = {}
    for (ts, te), (ms, me) in zip(*alignment.aligned):
        for k in range(min(te - ts, me - ms)):
            pocket_to_seq[ms + k] = ts + k
    idx = [pocket_to_seq.get(p) for p in (POS_LYS, POS_GLU, POS_PHE)]
    return None if any(i is None for i in idx) else tuple(idx)


_MOL2_RESIDUE = re.compile(r"^([A-Z0-9]{3})(-?\d+)[A-Z]?$")


def read_pocket_residues(mol2_path: str | Path) -> list[tuple[str, int]]:
    """``(residue name, author number)`` of a KLIFS pocket mol2, in pocket
    order (only resolved pocket residues are listed by KLIFS)."""
    residues, in_section = [], False
    for line in Path(mol2_path).read_text().splitlines():
        if line.startswith("@<TRIPOS>"):
            in_section = line.strip() == "@<TRIPOS>SUBSTRUCTURE"
            continue
        tokens = line.split()
        if in_section and len(tokens) >= 2 and tokens[0].isdigit():
            match = _MOL2_RESIDUE.match(tokens[1])
            if match:
                residues.append((match.group(1), int(match.group(2))))
    return residues


def locate_motif_by_numbering(
    pocket: str | None, pocket_residues: list[tuple[str, int]], structure
) -> tuple[int, int, int] | None:
    """0-based SEQRES indices of (β3-Lys, αC-Glu, DFG-Phe) from KLIFS's own
    residue numbering: non-gap pocket positions pair, in order, with the
    mol2 residues; each motif residue is found in the first chain by author
    number and must carry the same residue name. None if anything disagrees
    or a motif residue is unresolved."""
    if not isinstance(pocket, str) or len(pocket) != 85:
        return None
    positions = [i for i, c in enumerate(pocket) if c != "_"]
    if len(positions) != len(pocket_residues):
        return None
    numbering = dict(zip(positions, pocket_residues))
    by_author = {r.seqid.num: r for r in structure[0][0] if r.label_seq}
    indices = []
    for position in (POS_LYS, POS_GLU, POS_PHE):
        if position not in numbering:
            return None
        name, number = numbering[position]
        residue = by_author.get(number)
        if residue is None or residue.name != name:
            return None
        indices.append(residue.label_seq - 1)
    return tuple(indices)


def motif_features(structure, idx: tuple[int, int, int]) -> dict[str, float]:
    """Motif distances (Å) in the first chain; residues found by
    ``label_seq_id == index + 1``. Missing atoms give NaN."""
    by_seq = {r.label_seq: r for r in structure[0][0] if r.label_seq}

    def pos(index: int, atom: str):
        residue = by_seq.get(index + 1)
        found = residue.find_atom(atom, "*") if residue is not None else None
        return None if found is None else np.array([found.pos.x, found.pos.y, found.pos.z])

    def dist(a, b) -> float:
        return float("nan") if a is None or b is None else float(np.linalg.norm(a - b))

    lys, glu, phe = idx
    return {
        "phe_glu_ca": dist(pos(phe, "CA"), pos(glu, "CA")),
        "phe_lys_ca": dist(pos(phe, "CA"), pos(lys, "CA")),
        "phe_cz_glu_ca": dist(pos(phe, "CZ"), pos(glu, "CA")),
        "phe_cz_lys_ca": dist(pos(phe, "CZ"), pos(lys, "CA")),
        "lys_glu_ca": dist(pos(lys, "CA"), pos(glu, "CA")),
        "lys_nz_glu_cd": dist(pos(lys, "NZ"), pos(glu, "CD")),
    }


def _fit_tree(features: pd.DataFrame, labels: pd.Series, columns: list[str]) -> dict:
    from sklearn.tree import DecisionTreeClassifier

    known = labels.notna()
    tree = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=0)
    tree.fit(features.loc[known, columns].to_numpy(float), labels[known].astype(str))
    t = tree.tree_
    nodes = [
        {
            "feature": columns[t.feature[i]] if t.children_left[i] != -1 else None,
            "threshold": float(t.threshold[i]),
            "left": int(t.children_left[i]),
            "right": int(t.children_right[i]),
            "missing_left": bool(t.missing_go_to_left[i]),
            "label": str(tree.classes_[int(np.argmax(t.value[i][0]))]),
        }
        for i in range(t.node_count)
    ]
    return {"features": columns, "nodes": nodes}


def _predict_tree(tree: dict, row: pd.Series) -> str:
    nodes, i = tree["nodes"], 0
    while nodes[i]["left"] != -1:
        value = row[nodes[i]["feature"]]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            go_left = nodes[i]["missing_left"]
        else:
            go_left = value <= nodes[i]["threshold"]
        i = nodes[i]["left"] if go_left else nodes[i]["right"]
    return nodes[i]["label"]


class StateLabeler:
    """DFG and αC labels from motif distances (two depth-2 decision trees)."""

    def __init__(self, trees: dict | None = None):
        self.trees = trees or {}

    def fit(self, features: pd.DataFrame, dfg: pd.Series, ac: pd.Series) -> StateLabeler:
        """Fit on structures with published labels (``None`` labels are ignored)."""
        features = features.reset_index(drop=True)
        self.trees = {
            "dfg": _fit_tree(features, pd.Series(dfg).reset_index(drop=True), DFG_FEATURES),
            "ac_helix": _fit_tree(features, pd.Series(ac).reset_index(drop=True), AC_FEATURES),
        }
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Columns ``dfg`` and ``ac_helix`` (``"in"`` / ``"out"``), same index."""
        return pd.DataFrame(
            {
                name: [_predict_tree(tree, row) for _, row in features.iterrows()]
                for name, tree in self.trees.items()
            },
            index=features.index,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.trees, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> StateLabeler:
        return cls(json.loads(Path(path).read_text()))
