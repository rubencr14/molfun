"""
Real GPU tests for all fine-tuning strategies on OpenFold.

Verifies that gradients flow correctly and params actually change.

Requires:
    - OpenFold installed
    - Weights at ~/.molfun/weights/finetuning_ptm_2.pt
    - CUDA GPU

Run:
    pytest tests/training/test_strategies_real.py -v -s
"""

import pytest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from molfun.models.structure import MolfunStructureModel
from molfun.training import (
    HeadOnlyFinetune,
    LoRAFinetune,
    PartialFinetune,
    FullFinetune,
)
from molfun.helpers.training import unpack_batch as _unpack_batch, to_device as _to_device

WEIGHTS_PATH = Path.home() / ".molfun" / "weights" / "finetuning_ptm_2.pt"


def _openfold_available() -> bool:
    try:
        from openfold.model.model import AlphaFold
        return True
    except ImportError:
        return False


skip_no_openfold = pytest.mark.skipif(
    not _openfold_available(), reason="OpenFold not installed",
)
skip_no_weights = pytest.mark.skipif(
    not WEIGHTS_PATH.exists(), reason=f"Weights not at {WEIGHTS_PATH}",
)
skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available",
)

requires_real = pytest.mark.usefixtures()
all_skips = [skip_no_cuda, skip_no_openfold, skip_no_weights]


def _make_model():
    from openfold.config import model_config
    config = model_config("model_1_ptm")
    return MolfunStructureModel(
        "openfold", config=config, weights=str(WEIGHTS_PATH), device="cuda",
        head="affinity", head_config={"single_dim": 384, "hidden_dim": 64},
    )


def _make_loader(n=2, batch_size=1):
    from tests.models.test_openfold_real import DummyOpenFoldDataset, _collate
    ds = DummyOpenFoldDataset(n_samples=n, seq_len=16, n_msa=4)
    return DataLoader(ds, batch_size=batch_size, collate_fn=_collate)


def _check_grads(model, strategy) -> dict:
    """Run one manual step and return which param groups got gradients."""
    model.adapter.train()
    model.head.train()
    groups = strategy.param_groups(model)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)

    loader = _make_loader(n=1)
    batch_data = next(iter(loader))
    batch, targets, mask = _unpack_batch(batch_data)
    batch = _to_device(batch, model.device)
    targets = targets.to(model.device)

    optimizer.zero_grad()
    result = model.forward(batch, mask=mask)
    loss = model.head.loss(result["preds"], targets)["affinity_loss"]
    loss.backward()

    info = {}
    for gi, g in enumerate(groups):
        with_grad = sum(1 for p in g["params"] if p.grad is not None)
        nonzero = sum(
            1 for p in g["params"]
            if p.grad is not None and p.grad.abs().max() > 0
        )
        info[gi] = {"total": len(g["params"]), "with_grad": with_grad,
                     "nonzero": nonzero, "lr": g["lr"]}
    return info


@skip_no_cuda
@skip_no_openfold
@skip_no_weights
class TestHeadOnlyReal:

    def test_grads_only_head(self):
        model = _make_model()
        strategy = HeadOnlyFinetune(lr=1e-3, amp=False)
        strategy.setup(model)
        info = _check_grads(model, strategy)

        assert info[0]["nonzero"] == info[0]["total"]

    def test_fit_one_epoch(self):
        model = _make_model()
        strategy = HeadOnlyFinetune(lr=1e-3, amp=False, warmup_steps=1)
        loader = _make_loader()
        history = strategy.fit(model, loader, epochs=1)

        assert len(history) == 1
        assert history[0]["train_loss"] > 0


@skip_no_cuda
@skip_no_openfold
@skip_no_weights
class TestLoRAReal:

    def test_grads_flow_to_lora_and_head(self):
        model = _make_model()
        strategy = LoRAFinetune(
            rank=4, target_modules=["linear_q", "linear_v"], use_hf=False,
            lr_lora=1e-4, lr_head=1e-3, amp=False,
        )
        strategy.setup(model)
        info = _check_grads(model, strategy)

        head_group = info[0]
        lora_group = info[1]
        assert head_group["nonzero"] == head_group["total"]
        assert lora_group["nonzero"] > 0

    def test_fit_with_ema(self):
        model = _make_model()
        strategy = LoRAFinetune(
            rank=4, target_modules=["linear_q"], use_hf=False,
            lr_lora=1e-4, lr_head=1e-3, amp=False,
            ema_decay=0.99, warmup_steps=1,
        )
        loader = _make_loader()
        history = strategy.fit(model, loader, epochs=1)

        assert history[0]["train_loss"] > 0
        assert strategy.ema is not None

    def test_lora_params_change(self):
        model = _make_model()
        strategy = LoRAFinetune(
            rank=4, target_modules=["linear_q"], use_hf=False,
            lr_lora=1e-2, lr_head=1e-2, amp=False,
            scheduler="constant",
        )
        strategy.setup(model)
        lora_params = model._peft.trainable_parameters()
        before = {id(p): p.data.clone() for p in lora_params}

        loader = _make_loader()
        strategy.fit(model, loader, epochs=1)

        changed = sum(
            1 for p in lora_params
            if (before[id(p)] - p.data).abs().max() > 1e-8
        )
        assert changed > 0, f"0/{len(lora_params)} LoRA params changed"


@skip_no_cuda
@skip_no_openfold
@skip_no_weights
class TestPartialReal:

    def test_grads_flow_to_unfrozen_blocks(self):
        model = _make_model()
        strategy = PartialFinetune(
            unfreeze_last_n=4, lr_trunk=1e-5, lr_head=1e-3, amp=False,
        )
        strategy.setup(model)
        info = _check_grads(model, strategy)

        trunk_group = info[0]
        head_group = info[1]
        assert trunk_group["nonzero"] > 0
        assert head_group["nonzero"] == head_group["total"]

    def test_early_blocks_frozen(self):
        model = _make_model()
        strategy = PartialFinetune(unfreeze_last_n=4, amp=False)
        strategy.setup(model)

        blocks = model.adapter.get_trunk_blocks()
        for p in blocks[0].parameters():
            assert not p.requires_grad
        for p in blocks[-1].parameters():
            assert p.requires_grad

    def test_fit_one_epoch(self):
        model = _make_model()
        strategy = PartialFinetune(
            unfreeze_last_n=2, lr_trunk=1e-5, lr_head=1e-3,
            amp=False, warmup_steps=1,
        )
        loader = _make_loader()
        history = strategy.fit(model, loader, epochs=1)

        assert history[0]["train_loss"] > 0


@skip_no_cuda
@skip_no_openfold
@skip_no_weights
class TestFullReal:

    def test_grads_flow_everywhere(self):
        model = _make_model()
        strategy = FullFinetune(
            lr=1e-5, lr_head=1e-3, layer_lr_decay=0.9, amp=False,
        )
        strategy.setup(model)
        info = _check_grads(model, strategy)

        # Evoformer blocks (groups 1..48) should all have gradients
        for gi in range(1, 49):
            assert info[gi]["nonzero"] > 0, f"Block group {gi} has no gradients"

        # Head should have gradients
        head_gi = max(info.keys())
        assert info[head_gi]["nonzero"] == info[head_gi]["total"]

    def test_layer_lr_ordering(self):
        model = _make_model()
        strategy = FullFinetune(lr=1e-5, layer_lr_decay=0.9, lr_head=1e-3)
        strategy.setup(model)
        groups = strategy.param_groups(model)

        # Evoformer block LRs should increase (groups 1..48)
        block_lrs = [g["lr"] for g in groups[1:49]]
        for i in range(len(block_lrs) - 1):
            assert block_lrs[i] < block_lrs[i + 1]

        # Head (last) should be highest
        assert groups[-1]["lr"] == 1e-3
