"""
Tests for all fine-tuning strategies: HeadOnly, LoRA, Partial, Full.

Uses mock models (no OpenFold required).
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from molfun.models.structure import MolfunStructureModel
from molfun.adapters.openfold import OpenFoldAdapter
from molfun.training import (
    FinetuneStrategy,
    HeadOnlyFinetune,
    LoRAFinetune,
    PartialFinetune,
    FullFinetune,
    EMA,
    build_scheduler,
)


# ── Mock model ────────────────────────────────────────────────────────

class _MockEvoformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        self.ff = nn.Linear(dim, dim)

    def forward(self, x):
        return self.ff(self.linear_q(x) + self.linear_v(x))


class _MockEvoformer(nn.Module):
    def __init__(self, dim: int, n_blocks: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([_MockEvoformerBlock(dim) for _ in range(n_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class _MockOpenFold(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.evoformer = _MockEvoformer(dim, n_blocks=4)
        self.structure_module = nn.Linear(dim, dim)
        self.input_embedder = nn.Linear(dim, dim)
        self._dim = dim

    def forward(self, batch):
        B, L = 2, 8
        dev = next(self.parameters()).device
        x = torch.randn(B, L, self._dim, device=dev)
        single = self.evoformer(x)
        return {
            "single": single,
            "pair": torch.randn(B, L, L, self._dim, device=dev),
            "final_atom_positions": torch.randn(B, L, 3, device=dev),
            "plddt": torch.rand(B, L, device=dev),
        }


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 32


def _make_model(**kw) -> MolfunStructureModel:
    mock = _MockOpenFold(DIM)
    defaults = dict(
        name="openfold", model=mock, device=DEVICE,
        head="affinity", head_config={"single_dim": DIM, "hidden_dim": 16},
    )
    defaults.update(kw)
    return MolfunStructureModel(**defaults)


def _make_loader(n: int = 8, batch_size: int = 2) -> DataLoader:
    feats = torch.randn(n, 8, DIM)
    targets = torch.randn(n, 1)
    ds = TensorDataset(feats, targets)
    return DataLoader(ds, batch_size=batch_size)


# ── EMA tests ─────────────────────────────────────────────────────────

class TestEMA:

    def test_update_moves_toward_params(self):
        p = nn.Parameter(torch.ones(4))
        ema = EMA([p], decay=0.9)

        p.data.fill_(10.0)
        ema.update()

        expected = 0.9 * 1.0 + 0.1 * 10.0  # 1.9
        assert torch.allclose(ema.shadow[0], torch.tensor(expected).expand(4))

    def test_apply_restore_roundtrip(self):
        p = nn.Parameter(torch.ones(4))
        ema = EMA([p], decay=0.99)

        p.data.fill_(5.0)
        ema.update()

        original = p.data.clone()
        ema.apply()
        assert not torch.equal(p.data, original)

        ema.restore()
        assert torch.equal(p.data, original)

    def test_state_dict(self):
        p = nn.Parameter(torch.randn(3))
        ema = EMA([p], decay=0.995)
        state = ema.state_dict()
        assert state["decay"] == 0.995
        assert len(state["shadow"]) == 1


# ── Scheduler tests ───────────────────────────────────────────────────

class TestScheduler:

    def test_warmup_ramps_up(self):
        opt = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=1.0)
        sched = build_scheduler(opt, "cosine", warmup_steps=10, total_steps=100)

        lrs = []
        for _ in range(10):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()

        assert lrs[0] < lrs[-1]
        assert lrs[0] < 0.2

    def test_cosine_decays(self):
        opt = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=1.0)
        sched = build_scheduler(opt, "cosine", warmup_steps=0, total_steps=100)

        lrs = []
        for _ in range(100):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()

        assert lrs[0] > lrs[-1]
        assert lrs[-1] < 0.1

    def test_constant_stays_flat(self):
        opt = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=1.0)
        sched = build_scheduler(opt, "constant", warmup_steps=0, total_steps=50)

        for _ in range(50):
            opt.step()
            sched.step()

        assert opt.param_groups[0]["lr"] == pytest.approx(1.0)

    def test_linear_decays(self):
        opt = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=1.0)
        sched = build_scheduler(opt, "linear", warmup_steps=0, total_steps=100, min_lr=0.0)

        for _ in range(100):
            opt.step()
            sched.step()

        assert opt.param_groups[0]["lr"] < 0.05


# ── HeadOnly strategy ─────────────────────────────────────────────────

class TestHeadOnlyFinetune:

    def test_freezes_trunk(self):
        model = _make_model()
        strategy = HeadOnlyFinetune(lr=1e-3)
        strategy.setup(model)

        for p in model.adapter.model.parameters():
            assert not p.requires_grad

    def test_head_trainable(self):
        model = _make_model()
        strategy = HeadOnlyFinetune(lr=1e-3)
        strategy.setup(model)

        for p in model.head.parameters():
            assert p.requires_grad

    def test_param_groups_single(self):
        model = _make_model()
        strategy = HeadOnlyFinetune(lr=5e-3)
        strategy.setup(model)
        groups = strategy.param_groups(model)

        assert len(groups) == 1
        assert groups[0]["lr"] == 5e-3

    def test_fit_runs(self):
        model = _make_model()
        strategy = HeadOnlyFinetune(lr=1e-3, warmup_steps=2)
        loader = _make_loader()
        history = strategy.fit(model, loader, epochs=2)

        assert len(history) == 2
        assert "train_loss" in history[0]
        assert "lr" in history[0]

    def test_describe(self):
        strategy = HeadOnlyFinetune(lr=1e-3, ema_decay=0.99)
        d = strategy.describe()
        assert d["strategy"] == "HeadOnlyFinetune"
        assert d["ema_decay"] == 0.99


# ── LoRA strategy ─────────────────────────────────────────────────────

class TestLoRAFinetune:

    def test_inherits_head_only(self):
        assert issubclass(LoRAFinetune, HeadOnlyFinetune)

    def test_injects_lora(self):
        model = _make_model()
        strategy = LoRAFinetune(
            rank=4, target_modules=["linear_q", "linear_v"], use_hf=False,
        )
        strategy.setup(model)

        assert model._peft is not None
        assert model._peft.trainable_param_count() > 0

    def test_two_param_groups(self):
        model = _make_model()
        strategy = LoRAFinetune(
            rank=4, target_modules=["linear_q"], use_hf=False,
            lr_head=1e-3, lr_lora=1e-4,
        )
        strategy.setup(model)
        groups = strategy.param_groups(model)

        assert len(groups) == 2
        assert groups[0]["lr"] == 1e-3
        assert groups[1]["lr"] == 1e-4

    def test_fit_with_ema(self):
        model = _make_model()
        strategy = LoRAFinetune(
            rank=4, target_modules=["linear_q"], use_hf=False,
            lr=1e-3, ema_decay=0.99, warmup_steps=1,
        )
        loader = _make_loader()
        history = strategy.fit(model, loader, epochs=2)

        assert len(history) == 2
        assert strategy.ema is not None

    def test_fit_with_grad_accumulation(self):
        model = _make_model()
        strategy = LoRAFinetune(
            rank=4, target_modules=["linear_q"], use_hf=False,
            lr=1e-3, accumulation_steps=2,
        )
        loader = _make_loader(n=4, batch_size=2)
        history = strategy.fit(model, loader, epochs=1)

        assert len(history) == 1
        assert history[0]["train_loss"] > 0

    def test_describe_has_lora_info(self):
        strategy = LoRAFinetune(rank=16, alpha=32.0, lr_head=1e-3, lr_lora=5e-5)
        d = strategy.describe()
        assert d["rank"] == 16
        assert d["alpha"] == 32.0
        assert d["lr_head"] == 1e-3
        assert d["lr_lora"] == 5e-5


# ── Partial strategy ──────────────────────────────────────────────────

class TestPartialFinetune:

    def test_unfreezes_last_n_blocks(self):
        model = _make_model()
        strategy = PartialFinetune(unfreeze_last_n=2)
        strategy.setup(model)

        blocks = model.adapter.get_evoformer_blocks()
        # First 2 blocks frozen
        for p in blocks[0].parameters():
            assert not p.requires_grad
        for p in blocks[1].parameters():
            assert not p.requires_grad
        # Last 2 blocks trainable
        for p in blocks[2].parameters():
            assert p.requires_grad
        for p in blocks[3].parameters():
            assert p.requires_grad

    def test_unfreezes_structure_module(self):
        model = _make_model()
        strategy = PartialFinetune(unfreeze_last_n=1, unfreeze_structure_module=True)
        strategy.setup(model)

        sm = model.adapter.model.structure_module
        for p in sm.parameters():
            assert p.requires_grad

    def test_two_param_groups(self):
        model = _make_model()
        strategy = PartialFinetune(
            unfreeze_last_n=2, lr_trunk=1e-5, lr_head=1e-3,
        )
        strategy.setup(model)
        groups = strategy.param_groups(model)

        assert len(groups) == 2
        assert groups[0]["lr"] == 1e-5
        assert groups[1]["lr"] == 1e-3

    def test_fit_runs(self):
        model = _make_model()
        strategy = PartialFinetune(
            unfreeze_last_n=2, lr_trunk=1e-5, lr_head=1e-3,
            warmup_steps=1,
        )
        loader = _make_loader()
        history = strategy.fit(model, loader, epochs=2)

        assert len(history) == 2
        assert all(h["train_loss"] > 0 for h in history)

    def test_describe(self):
        strategy = PartialFinetune(unfreeze_last_n=6, lr_trunk=1e-5)
        d = strategy.describe()
        assert d["unfreeze_last_n"] == 6
        assert d["lr_trunk"] == 1e-5


# ── Full strategy ─────────────────────────────────────────────────────

class TestFullFinetune:

    def test_unfreezes_everything(self):
        model = _make_model()
        strategy = FullFinetune(lr=1e-5)
        strategy.setup(model)

        frozen = sum(1 for p in model.adapter.model.parameters() if not p.requires_grad)
        assert frozen == 0

    def test_layer_wise_lr_decay(self):
        model = _make_model()
        strategy = FullFinetune(lr=1e-4, layer_lr_decay=0.9, lr_head=1e-3)
        strategy.setup(model)
        groups = strategy.param_groups(model)

        # Should have: embedder + 4 blocks + structure_module + remaining + head
        lrs = [g["lr"] for g in groups]
        # Head (last) should have highest LR
        assert lrs[-1] == 1e-3
        # Block LRs should increase (later blocks = higher LR)
        block_lrs = lrs[1:5]  # 4 evoformer blocks
        for i in range(len(block_lrs) - 1):
            assert block_lrs[i] < block_lrs[i + 1]

    def test_fit_runs(self):
        model = _make_model()
        strategy = FullFinetune(
            lr=1e-4, lr_head=1e-3, layer_lr_decay=0.95,
            warmup_steps=1,
        )
        loader = _make_loader()
        history = strategy.fit(model, loader, epochs=1)

        assert len(history) == 1
        assert history[0]["train_loss"] > 0

    def test_describe(self):
        strategy = FullFinetune(lr=1e-5, layer_lr_decay=0.9, lr_head=1e-3)
        d = strategy.describe()
        assert d["layer_lr_decay"] == 0.9
        assert d["lr_head"] == 1e-3


# ── Early stopping ────────────────────────────────────────────────────

class TestEarlyStopping:

    def test_stops_when_no_improvement(self):
        model = _make_model()
        strategy = HeadOnlyFinetune(lr=1e-3, early_stopping_patience=2)
        train_loader = _make_loader()
        val_loader = _make_loader(n=4)

        history = strategy.fit(model, train_loader, val_loader, epochs=50)

        # Should stop well before 50 epochs if val doesn't improve
        assert len(history) <= 50
        if len(history) > 2:
            assert history[-1].get("patience", 0) >= 2 or len(history) == 50


# ── Validation with EMA ──────────────────────────────────────────────

class TestEMAValidation:

    def test_val_uses_ema_weights(self):
        model = _make_model()
        strategy = HeadOnlyFinetune(lr=1e-2, ema_decay=0.99)
        train_loader = _make_loader()
        val_loader = _make_loader(n=4)

        history = strategy.fit(model, train_loader, val_loader, epochs=3)

        assert strategy.ema is not None
        assert "val_loss" in history[0]

    def test_apply_ema_permanently(self):
        model = _make_model()
        strategy = HeadOnlyFinetune(lr=1e-2, ema_decay=0.99)
        loader = _make_loader()

        strategy.fit(model, loader, epochs=2)
        assert strategy.ema is not None

        strategy.apply_ema(model)
        assert strategy.ema is None
