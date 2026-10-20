"""Structure accuracy metrics on CA coordinates."""

import gemmi
import numpy as np
import pytest

from molfun.core.types import SampledStructure
from molfun.eval.structure import ca_rmsd, lddt_ca, paired_ca, tm_score


def _helix(n=40):
    """An ideal alpha-helix CA trace (1.5 Å rise, 100° per residue, 2.3 Å radius)."""
    t = np.radians(100.0) * np.arange(n)
    return np.stack([2.3 * np.cos(t), 2.3 * np.sin(t), 1.5 * np.arange(n)], axis=1)


def _rotation(deg=37.0):
    a = np.radians(deg)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return rz @ rx


def test_rmsd_zero_for_identical():
    x = _helix()

    assert ca_rmsd(x, x) == pytest.approx(0.0, abs=1e-6)


def test_rmsd_invariant_to_rotation_and_translation():
    x = _helix()
    moved = x @ _rotation().T + np.array([5.0, -3.0, 12.0])

    assert ca_rmsd(moved, x) == pytest.approx(0.0, abs=1e-5)


def test_rmsd_of_known_displacement():
    x = _helix()
    shifted = x.copy()
    shifted[:, 0] += np.where(np.arange(len(x)) % 2 == 0, 1.0, -1.0)  # ±1 Å alternating

    assert ca_rmsd(shifted, x) == pytest.approx(1.0, abs=0.1)


def test_lddt_one_for_identical():
    x = _helix()

    assert lddt_ca(x, x) == pytest.approx(1.0)


def test_lddt_drops_with_noise():
    rng = np.random.default_rng(0)
    x = _helix()
    noisy = x + rng.normal(scale=3.0, size=x.shape)

    score = lddt_ca(noisy, x)

    assert 0.0 < score < 0.8


def test_lddt_ignores_rigid_motion():
    x = _helix()

    assert lddt_ca(x @ _rotation().T + 10.0, x) == pytest.approx(1.0)


def test_tm_score_identical_is_one():
    x = _helix(60)

    assert tm_score(x @ _rotation().T, x) == pytest.approx(1.0, abs=1e-3)


def _akt1_ca():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "fixtures" / "kinase" / "3mvh_A.cif.gz"
    chain = gemmi.read_structure(str(path))[0][0]
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for r in chain for a in r if a.name == "CA"])


def test_tm_score_uses_the_residue_pairing():
    """A prediction shifted by one residue along the chain is a wrong
    prediction; TM-align would re-align it and score ~1."""
    from tmtools import tm_align

    ca = _akt1_ca()
    shifted, ref = ca[1:], ca[:-1]  # residue i predicted at residue i+1's place
    realigned = tm_align(shifted, ref, "A" * len(ref), "A" * len(ref)).tm_norm_chain2

    assert realigned > 0.95
    assert tm_score(shifted, ref) < 0.75
    assert tm_score(ca, ca) == pytest.approx(1.0)


def test_tm_score_partial_match_scores_between():
    """Half the chain right, half displaced by 10 Å. Superposing the right half
    alone gives 0.5 + 0.5 / (1 + (10 / d0)²) with d0 = 1.24·(L−15)^⅓ − 1.8, so
    the TM-score is at least that, and clearly below a correct prediction.
    (A real kinase chain: an ideal helix would partly re-superpose by symmetry.)"""
    ca = _akt1_ca()
    pred = ca.copy()
    pred[len(ca) // 2 :] += np.array([10.0, 0.0, 0.0])
    d0 = 1.24 * (len(ca) - 15) ** (1 / 3) - 1.8
    lower = 0.5 + 0.5 / (1 + (10 / d0) ** 2)

    assert lower - 1e-6 <= tm_score(pred, ca) < 0.9


def _ca_mmcif(coords: np.ndarray, present: list[int]) -> str:
    """mmCIF with one chain A of ALA residues; residue i has label_seq i+1."""
    st = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for i in present:
        res = gemmi.Residue()
        res.name = "ALA"
        res.seqid = gemmi.SeqId(i + 1, " ")
        res.label_seq = i + 1
        res.entity_type = gemmi.EntityType.Polymer
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(*coords[i])
        res.add_atom(atom)
        chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    return st.make_mmcif_document().as_string()


def test_paired_ca_skips_unresolved(tmp_path):
    x = _helix(20)
    resolved = [i for i in range(20) if not 5 <= i <= 9]
    ref = tmp_path / "ref.cif"
    ref.write_text(_ca_mmcif(x, resolved))
    pred = SampledStructure(sequence="A" * 20, mmcif=_ca_mmcif(x + 1.0, list(range(20))))

    p, r = paired_ca(pred, ref, "A")

    assert p.shape == r.shape == (15, 3)
    assert np.allclose(r, x[resolved])
    assert np.allclose(p, x[resolved] + 1.0)
