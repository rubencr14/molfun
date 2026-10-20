"""Kinase motif location and the geometric DFG / αC state labeler."""

from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

from molfun.families.kinase.states import (
    StateLabeler,
    locate_motif,
    locate_motif_by_numbering,
    read_pocket_residues,
)

KINASE = Path(__file__).resolve().parents[1] / "fixtures" / "kinase"

# AKT1, PDB 3MVH chain A: KLIFS pocket and SEQRES.
AKT1_POCKET = (
    "KLLGKGTFGKVILYAMKILHTLTENRVLQNSRPFLTALKYSCFVMEYANGGELFFHLSRLHSEKNVVYRDLKLENLMLITDFGLC"
)
AKT1_SEQRES = (
    "GAMDPRVTMNEFEYLKLLGKGTFGKVILVKEKATGRYYAMKILKKEVIVAKDEVAHTLTENRVLQNSRHPFLTALKYSFQTHDRLC"
    "FVMEYANGGELFFHLSRERVFSEDRARFYGAEIVSALDYLHSEKNVVYRDLKLENLMLDKDGHIKITDFGLCKEGIKDGATMKTFC"
    "GTPEYLAPEVLEDNDYGRAVDWWGLGVVMYEMMCGRLPFYNQDHEKLFELILMEEIRFPRTLGPEAKSLLSGLLKKDPKQRLGGG"
    "SEDAKEIMQHRFFAGIVWQHVYEKKLSPPFKPQVTSETDTRYFDEEFTAQMITITPPDQDDSMECVDSERRPHFPQFDYSASSTA"
)


def test_locate_motif_on_known_pocket():
    lys, glu, phe = locate_motif(AKT1_POCKET, AKT1_SEQRES)

    assert AKT1_SEQRES[lys] == "K"  # beta3 Lys (K179 in AKT1)
    assert AKT1_SEQRES[glu] == "E"  # alphaC Glu (E198)
    assert AKT1_SEQRES[phe - 1 : phe + 2] == "DFG"
    assert glu - lys == 19  # K179 → E198
    assert phe - lys == 114  # K179 → F293


def test_locate_motif_rejects_bad_pocket():
    assert locate_motif("SHORT", AKT1_SEQRES) is None
    assert locate_motif(None, AKT1_SEQRES) is None


def test_read_pocket_residues_ignores_trailing_comment():
    residues = read_pocket_residues(KINASE / "2538_pocket.mol2")

    assert len(residues) == 85
    assert residues[0] == ("LYS", 154)
    assert residues[16] == ("LYS", 179)


def _akt1_chain():
    return gemmi.read_structure(str(KINASE / "3mvh_A.cif.gz"))


def _residue_at(structure, index):
    return next(r for r in structure[0][0] if r.label_seq == index + 1)


def test_locate_motif_by_numbering_on_akt1():
    structure = _akt1_chain()
    residues = read_pocket_residues(KINASE / "2538_pocket.mol2")

    idx = locate_motif_by_numbering(AKT1_POCKET, residues, structure)

    assert [_residue_at(structure, i).seqid.num for i in idx] == [179, 198, 293]
    lys, glu, phe = idx
    assert AKT1_SEQRES[lys] == "K" and AKT1_SEQRES[glu] == "E"
    assert AKT1_SEQRES[phe - 1 : phe + 2] == "DFG"


def test_locate_motif_by_numbering_follows_pocket_gaps():
    structure = _akt1_chain()
    residues = read_pocket_residues(KINASE / "2538_pocket.mol2")
    gapped = "_" + AKT1_POCKET[1:]  # position 1 unresolved → absent from the mol2

    idx = locate_motif_by_numbering(gapped, residues[1:], structure)

    assert [_residue_at(structure, i).seqid.num for i in idx] == [179, 198, 293]


def test_locate_motif_by_numbering_rejects_count_mismatch():
    structure = _akt1_chain()
    residues = read_pocket_residues(KINASE / "2538_pocket.mol2")

    assert locate_motif_by_numbering(AKT1_POCKET, residues[1:], structure) is None


def test_locate_motif_by_numbering_rejects_name_mismatch():
    structure = _akt1_chain()
    residues = read_pocket_residues(KINASE / "2538_pocket.mol2")
    residues[16] = ("ALA", 179)  # numbering points at a residue with another name

    assert locate_motif_by_numbering(AKT1_POCKET, residues, structure) is None


def _synthetic_features(n=40, seed=0):
    """DFG-in: Phe–Glu ≈ 6 Å; DFG-out: ≈ 11 Å. αC-in: Lys–Glu ≈ 4 Å side chains; out ≈ 12 Å."""
    rng = np.random.default_rng(seed)
    dfg = np.array(["in", "out"] * (n // 2), dtype=object)
    ac = np.array(["in"] * (n // 2) + ["out"] * (n // 2), dtype=object)
    feats = pd.DataFrame(
        {
            "phe_glu_ca": np.where(dfg == "in", 6.0, 11.0) + rng.normal(0, 0.5, n),
            "phe_lys_ca": np.where(dfg == "in", 12.0, 10.0) + rng.normal(0, 0.5, n),
            "phe_cz_glu_ca": np.where(dfg == "in", 7.0, 13.0) + rng.normal(0, 0.5, n),
            "phe_cz_lys_ca": np.where(dfg == "in", 13.0, 9.0) + rng.normal(0, 0.5, n),
            "lys_glu_ca": np.where(ac == "in", 13.0, 17.0) + rng.normal(0, 0.5, n),
            "lys_nz_glu_cd": np.where(ac == "in", 4.0, 12.0) + rng.normal(0, 0.5, n),
        }
    )
    return feats, pd.Series(dfg), pd.Series(ac)


def test_labeler_separates_synthetic_states():
    feats, dfg, ac = _synthetic_features()

    pred = StateLabeler().fit(feats, dfg, ac).predict(feats)

    assert (pred["dfg"].values == dfg.values).mean() == 1.0
    assert (pred["ac_helix"].values == ac.values).mean() == 1.0


def test_labeler_handles_missing_side_chain_atoms():
    feats, dfg, ac = _synthetic_features()
    labeler = StateLabeler().fit(feats, dfg, ac)
    feats_missing = feats.copy()
    feats_missing[["phe_cz_glu_ca", "phe_cz_lys_ca", "lys_nz_glu_cd"]] = np.nan

    pred = labeler.predict(feats_missing)

    assert pred["dfg"].isin(["in", "out"]).all()
    assert pred["ac_helix"].isin(["in", "out"]).all()


def test_labeler_roundtrip_json(tmp_path):
    feats, dfg, ac = _synthetic_features(seed=1)
    labeler = StateLabeler().fit(feats, dfg, ac)
    labeler.save(tmp_path / "labeler.json")

    loaded = StateLabeler.load(tmp_path / "labeler.json")

    pd.testing.assert_frame_equal(loaded.predict(feats), labeler.predict(feats))
