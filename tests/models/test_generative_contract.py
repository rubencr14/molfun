"""Tests for the generative-model contract: losses via TrunkOutput.extra,
the parameter-free generative head, sampled structures, and adapter registration."""

import gemmi
import numpy as np
import pytest
import torch

import molfun.models.structure as structure_mod
from molfun.core.types import SampledStructure, TrunkOutput
from molfun.heads.generative import GenerativeLossHead


def test_trunk_output_without_single_repr():
    out = TrunkOutput(extra={"losses": {}})

    assert out.single_repr is None


def test_generative_head_returns_total_first():
    a, b = torch.tensor(1.0), torch.tensor(3.0)
    out = TrunkOutput(extra={"losses": {"flow_mse": a, "total": b}})
    head = GenerativeLossHead()

    losses = head.loss(head(out))

    assert list(losses)[0] == "total"
    assert losses["total"].item() == 3.0
    assert losses["flow_mse"].item() == 1.0


def test_generative_head_missing_losses_raises():
    head = GenerativeLossHead()

    with pytest.raises(KeyError, match="losses"):
        head(TrunkOutput(extra={}))


def test_generative_head_registered():
    assert structure_mod.HEAD_REGISTRY["generative"] is GenerativeLossHead


def _three_residue_mmcif(missing_ca_at: int) -> str:
    st = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for i, name in enumerate(["GLY", "ALA", "SER"], start=1):
        res = gemmi.Residue()
        res.name = name
        res.seqid = gemmi.SeqId(i, " ")
        res.label_seq = i
        res.entity_type = gemmi.EntityType.Polymer
        atom = gemmi.Atom()
        atom.name = "N" if i == missing_ca_at else "CA"
        atom.element = gemmi.Element("N" if i == missing_ca_at else "C")
        atom.pos = gemmi.Position(float(i), 0.0, 0.0)
        res.add_atom(atom)
        chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    return st.make_mmcif_document().as_string()


def test_sampled_structure_ca_coords_nan_for_missing():
    sample = SampledStructure(sequence="GAS", mmcif=_three_residue_mmcif(missing_ca_at=2))

    ca = sample.ca_coords()

    assert ca.shape == (3, 3)
    assert np.allclose(ca[0], [1.0, 0.0, 0.0])
    assert np.isnan(ca[1]).all()
    assert np.allclose(ca[2], [3.0, 0.0, 0.0])


def test_register_adapters_survives_failing_backend(monkeypatch):
    monkeypatch.setattr(structure_mod, "ADAPTER_REGISTRY", {})
    monkeypatch.setattr(
        structure_mod,
        "_BUILTIN_ADAPTERS",
        {
            "broken": "molfun.backends.does_not_exist:Nope",
            "openfold": "molfun.backends.openfold.adapter:OpenFoldAdapter",
        },
    )

    structure_mod._register_adapters()

    assert "openfold" in structure_mod.ADAPTER_REGISTRY
    assert "broken" not in structure_mod.ADAPTER_REGISTRY


def test_register_adapters_after_custom_registration(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(structure_mod, "ADAPTER_REGISTRY", {"custom": sentinel})
    monkeypatch.setattr(
        structure_mod,
        "_BUILTIN_ADAPTERS",
        {"openfold": "molfun.backends.openfold.adapter:OpenFoldAdapter"},
    )

    structure_mod._register_adapters()

    assert structure_mod.ADAPTER_REGISTRY["custom"] is sentinel
    assert "openfold" in structure_mod.ADAPTER_REGISTRY
