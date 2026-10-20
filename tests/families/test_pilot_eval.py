"""Per-target pilot evaluation, checked on a model that returns the
experimental structure itself (no GPU)."""

import gzip
import shutil
from pathlib import Path

import gemmi
import pandas as pd
import pytest

from molfun.core.types import SampledStructure
from molfun.families.kinase.fit_labeler import chain_seqres
from molfun.families.kinase.pilot import evaluate_target
from molfun.families.kinase.states import StateLabeler
from tests.families.test_states import AKT1_POCKET, _synthetic_features

KINASE = Path(__file__).resolve().parents[1] / "fixtures" / "kinase"


class _EchoModel:
    """Returns the reference chain as every sample; records the call."""

    def __init__(self, mmcif: str):
        self.mmcif = mmcif
        self.calls = []

    def sample(self, sequence, **kwargs):
        self.calls.append((sequence, kwargs))
        return [SampledStructure(sequence, self.mmcif) for _ in range(kwargs["num_samples"])]


@pytest.fixture
def akt1(tmp_path):
    chains, pockets = tmp_path / "chains", tmp_path / "pockets"
    chains.mkdir()
    pockets.mkdir()
    with gzip.open(KINASE / "3mvh_A.cif.gz", "rb") as src, open(chains / "2538.cif", "wb") as dst:
        shutil.copyfileobj(src, dst)
    shutil.copy(KINASE / "2538_pocket.mol2", pockets / "2538.mol2")
    structure = gemmi.read_structure(str(chains / "2538.cif"))
    target = pd.Series(
        {
            "structure_id": "2538",
            "uniprot": "P31749",
            "chain_cif": str(chains / "2538.cif"),
            "chain": "A",
            "seqres": chain_seqres(structure),
            "pocket": AKT1_POCKET,
            "dfg": "in",
            "ac_helix": "in",
        }
    )
    mmcif = structure.make_mmcif_document().as_string()
    return target, pockets, mmcif


def test_evaluate_target_on_reference_itself(akt1):
    target, pockets, mmcif = akt1
    feats, dfg, ac = _synthetic_features()
    labeler = StateLabeler().fit(feats, dfg, ac)
    model = _EchoModel(mmcif)

    row = evaluate_target(
        model, target, labeler, pockets, num_samples=3, num_steps=7, tau=0.1, lddt_threshold=0.7
    )

    assert model.calls[0][0] == target["seqres"]  # sampled from the sequence only
    assert model.calls[0][1]["num_steps"] == 7
    assert row["n_pairs"] > 250
    assert row["tm_mean"] == pytest.approx(1.0, abs=1e-3)
    assert row["lddt_mean"] == pytest.approx(1.0)
    assert row["rmsd_mean"] == pytest.approx(0.0, abs=1e-3)
    assert row["state_top1_correct"] is True
    assert row["state_recovered"] is True
    assert row["sampled_states"] == ["in/in"] * 3
