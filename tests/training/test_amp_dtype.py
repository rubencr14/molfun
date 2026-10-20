"""Tests for the training loop's mixed-precision dtype option, using a toy
generative adapter that reports its losses through TrunkOutput.extra."""

import math

import pytest
import torch
from torch.utils.data import DataLoader

import molfun.models.structure as structure_mod
from molfun.models.structure import MolfunStructureModel
from molfun.training import FullFinetune
from tests.toy_adapters import ToyGenerativeAdapter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def toy_model(monkeypatch):
    registry = dict(structure_mod.ADAPTER_REGISTRY)
    registry["toy_generative"] = ToyGenerativeAdapter
    monkeypatch.setattr(structure_mod, "ADAPTER_REGISTRY", registry)
    return MolfunStructureModel(name="toy_generative", device=DEVICE, head="generative")


def _loader(n=6):
    data = [{"x": torch.randn(8, 4)} for _ in range(n)]
    return DataLoader(data, batch_size=1)


def test_bf16_disables_grad_scaler(toy_model):
    strategy = FullFinetune(lr=1e-3, layer_lr_decay=1.0, amp_dtype="bfloat16")

    scaler = strategy._make_scaler(toy_model)

    assert scaler.is_enabled() is False


def test_bf16_training_reduces_loss(toy_model):
    torch.manual_seed(0)
    strategy = FullFinetune(
        lr=5e-2, layer_lr_decay=1.0, warmup_steps=0, scheduler="constant", amp_dtype="bfloat16"
    )

    history = strategy.fit(toy_model, _loader(), epochs=3)

    losses = [h["train_loss"] for h in history]
    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < losses[0]


def test_unknown_amp_dtype_rejected():
    with pytest.raises(ValueError, match="amp_dtype"):
        FullFinetune(lr=1e-3, amp_dtype="int8")
