"""The KLIFS/KinCoRe kinase table built by the nomosis-kinases project.

``master.csv`` has one row per KLIFS structure (PDB entry + chain + altloc)
with KLIFS conformation labels. The pilot state is DFG × αC:
KLIFS ``out-like`` counts as ``out`` and ``na`` becomes ``None``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_COLUMNS = {
    "structure_ID": "structure_id",
    "pdb": "pdb",
    "chain": "chain",
    "alt": "alt",
    "uniprot": "uniprot",
    "kinase": "kinase",
    "family": "family",
    "group": "group",
    "species": "species",
    "pocket": "pocket",
    "resolution": "resolution",
    "DFG": "dfg",
    "aC_helix": "ac_helix",
}
_LABELS = {"in": "in", "out": "out", "out-like": "out"}


def _label(value) -> str | None:
    return _LABELS.get(value) if isinstance(value, str) else None


def _text(value) -> str | None:
    return value if isinstance(value, str) and value else None


def load_klifs_table(master_csv: str | Path) -> pd.DataFrame:
    """Load the curated KLIFS rows with normalised columns and labels.

    Returns columns ``structure_id`` (str), ``pdb``, ``chain``, ``alt`` (str or
    None), ``uniprot``, ``kinase``, ``family``, ``group``, ``species``,
    ``pocket`` (85-residue KLIFS pocket), ``resolution``, ``dfg`` and
    ``ac_helix`` (``"in"``, ``"out"`` or ``None``).
    """
    raw = pd.read_csv(master_csv, low_memory=False)
    raw = raw[raw["curated"].astype(bool)]
    table = raw[list(_COLUMNS)].rename(columns=_COLUMNS).reset_index(drop=True)
    table["structure_id"] = table["structure_id"].astype(str)
    table["pdb"] = table["pdb"].str.lower()
    table["alt"] = table["alt"].map(_text).astype(object)
    table["dfg"] = table["dfg"].map(_label).astype(object)
    table["ac_helix"] = table["ac_helix"].map(_label).astype(object)
    return table
