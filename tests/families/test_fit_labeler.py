"""Motif features of real chains, located through KLIFS pocket numbering."""

import gzip
import shutil
from pathlib import Path

import pandas as pd

from molfun.families.kinase.fit_labeler import compute_features
from tests.families.test_states import AKT1_POCKET

KINASE = Path(__file__).resolve().parents[1] / "fixtures" / "kinase"


def test_heldout_ids_come_from_the_saved_list(tmp_path):
    """Not rebuilt as extracted − train − dropped: stale or excluded chain
    files on disk must not join the held-out set."""
    import json

    import pytest

    from molfun.families.kinase.fit_labeler import heldout_ids

    (tmp_path / "chains").mkdir()
    for sid in ("1", "2", "3", "stale"):
        (tmp_path / "chains" / f"{sid}.cif").write_text("")
    (tmp_path / "train_ids.json").write_text(json.dumps(["1"]))
    (tmp_path / "heldout_ids.json").write_text(json.dumps(["2", "3"]))

    assert heldout_ids(tmp_path) == {"2", "3"}
    (tmp_path / "heldout_ids.json").unlink()
    with pytest.raises(FileNotFoundError, match="heldout_ids.json"):
        heldout_ids(tmp_path)


def test_compute_features_on_akt1_dfg_in_ac_in(tmp_path):
    chains, pockets = tmp_path / "chains", tmp_path / "pockets"
    chains.mkdir()
    pockets.mkdir()
    with gzip.open(KINASE / "3mvh_A.cif.gz", "rb") as src, open(chains / "2538.cif", "wb") as dst:
        shutil.copyfileobj(src, dst)
    shutil.copy(KINASE / "2538_pocket.mol2", pockets / "2538.mol2")
    rows = pd.DataFrame({"structure_id": ["2538", "9999"], "pocket": [AKT1_POCKET, AKT1_POCKET]})

    feats = compute_features(rows, chains, pockets, workers=1)

    assert list(feats["structure_id"]) == ["2538"]  # 9999 has no files → left out
    row = feats.iloc[0]
    assert row["phe_glu_ca"] < 9.0  # DFG-in: Phe near the αC Glu
    assert row["lys_nz_glu_cd"] < 5.0  # αC-in: Lys–Glu salt bridge
