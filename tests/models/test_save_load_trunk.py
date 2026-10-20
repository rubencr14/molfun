"""save()/load() must keep trunk weights of full (non-PEFT) fine-tunes."""

import pytest
import torch

import molfun.models.structure as structure_mod
from molfun.models.structure import MolfunStructureModel
from tests.toy_adapters import RenamedToyGenerativeAdapter, ToyGenerativeAdapter


@pytest.fixture
def make_model(monkeypatch):
    registry = dict(structure_mod.ADAPTER_REGISTRY)
    registry["toy_generative"] = ToyGenerativeAdapter
    registry["toy_renamed"] = RenamedToyGenerativeAdapter
    monkeypatch.setattr(structure_mod, "ADAPTER_REGISTRY", registry)

    def make(name="toy_generative"):
        return MolfunStructureModel(name=name, device="cpu", head="generative")

    return make


def test_full_finetune_roundtrip(make_model, tmp_path):
    torch.manual_seed(0)
    trained = make_model()
    with torch.no_grad():
        for p in trained.adapter.parameters():
            p.add_(1.0)
    trained.save(str(tmp_path / "ckpt"))

    torch.manual_seed(1)
    fresh = make_model()
    fresh.load(str(tmp_path / "ckpt"))

    assert (tmp_path / "ckpt" / "trunk.pt").exists()
    for (name, a), (_, b) in zip(
        trained.adapter.named_parameters(), fresh.adapter.named_parameters()
    ):
        assert torch.equal(a, b), name


def test_load_reports_key_mismatch(make_model, tmp_path):
    make_model().save(str(tmp_path / "ckpt"))
    other = make_model("toy_renamed")

    with pytest.raises(RuntimeError, match="blocks.0.weight"):
        other.load(str(tmp_path / "ckpt"))


def test_frozen_trunk_is_not_saved(make_model, tmp_path):
    model = make_model()
    model.adapter.freeze_trunk()

    model.save(str(tmp_path / "ckpt"))

    assert not (tmp_path / "ckpt" / "trunk.pt").exists()
