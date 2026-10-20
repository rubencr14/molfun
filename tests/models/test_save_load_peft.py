"""A LoRA checkpoint must load into a fresh model (no PEFT set up yet)."""

import pytest
import torch

import molfun.models.structure as structure_mod
from molfun.models.structure import MolfunStructureModel
from molfun.training import LoRAFinetune
from tests.toy_adapters import ToyGenerativeAdapter


@pytest.fixture
def make_model(monkeypatch):
    registry = dict(structure_mod.ADAPTER_REGISTRY)
    registry["toy_generative"] = ToyGenerativeAdapter
    monkeypatch.setattr(structure_mod, "ADAPTER_REGISTRY", registry)
    return lambda: MolfunStructureModel(name="toy_generative", device="cpu", head="generative")


def _trunk_output(model, x):
    with torch.no_grad():
        for block in model.adapter.get_trunk_blocks():
            x = block(x)
    return x


@pytest.mark.parametrize("use_hf", [True, False])
def test_lora_checkpoint_loads_into_fresh_model(make_model, tmp_path, use_hf):
    if use_hf:
        pytest.importorskip("peft")
    torch.manual_seed(0)
    trained = make_model()
    LoRAFinetune(rank=2, alpha=4.0, target_modules=["0", "1"], use_hf=use_hf).setup(trained)
    with torch.no_grad():
        for p in trained._peft.trainable_parameters():
            p.add_(0.5)
    x = torch.randn(3, 4)
    expected = _trunk_output(trained, x)
    trained.save(str(tmp_path / "ckpt"))

    torch.manual_seed(0)  # same base weights, as from_pretrained would give
    fresh = make_model()
    base_output = _trunk_output(fresh, x)
    fresh.load(str(tmp_path / "ckpt"))

    assert fresh._peft is not None
    assert torch.allclose(_trunk_output(fresh, x), expected, atol=1e-6)
    assert not torch.allclose(base_output, expected, atol=1e-3)


def test_peft_checkpoint_without_config_is_rejected(make_model, tmp_path):
    trained = make_model()
    LoRAFinetune(rank=2, target_modules=["0", "1"], use_hf=False).setup(trained)
    trained.save(str(tmp_path / "ckpt"))
    (tmp_path / "ckpt" / "peft_config.json").unlink()

    with pytest.raises(RuntimeError, match="peft_config"):
        make_model().load(str(tmp_path / "ckpt"))
