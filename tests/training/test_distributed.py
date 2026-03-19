"""
Tests for distributed training strategies and gradient checkpointing.

These tests run on CPU / single-GPU and verify the interfaces, wrapping
logic, and checkpointing without requiring a multi-GPU setup.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from molfun.training.distributed import (
    BaseDistributedStrategy,
    DDPStrategy,
    FSDPStrategy,
)
from molfun.training.checkpointing import (
    apply_gradient_checkpointing,
    remove_gradient_checkpointing,
    estimate_memory_savings,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

class SimpleBlock(nn.Module):
    """A named 'block' so auto-detection picks it up."""

    def __init__(self, dim=16):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.linear(x))


class SimpleModel(nn.Module):
    def __init__(self, dim=16, n_blocks=4):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([SimpleBlock(dim) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def _make_loader(n=20, dim=16, batch_size=4):
    X = torch.randn(n, dim)
    y = torch.randn(n, 1)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


# ------------------------------------------------------------------
# DDPStrategy tests (interface-level, no multi-GPU needed)
# ------------------------------------------------------------------

class TestDDPStrategy:
    def test_is_base_strategy(self):
        ddp = DDPStrategy()
        assert isinstance(ddp, BaseDistributedStrategy)

    def test_default_config(self):
        ddp = DDPStrategy()
        assert ddp._backend == "nccl"
        assert ddp._find_unused is False
        assert ddp._bucket_view is True

    def test_custom_config(self):
        ddp = DDPStrategy(
            backend="gloo",
            find_unused_parameters=True,
            gradient_as_bucket_view=False,
            static_graph=True,
        )
        assert ddp._backend == "gloo"
        assert ddp._find_unused is True
        assert ddp._static_graph is True

    def test_is_main_process_default(self):
        ddp = DDPStrategy()
        assert ddp.is_main_process  # rank 0 by default

    def test_local_rank_default(self):
        ddp = DDPStrategy()
        assert ddp.local_rank == 0

    def test_wrap_loader_creates_distributed_sampler(self):
        ddp = DDPStrategy()
        loader = _make_loader()
        wrapped = ddp.wrap_loader(loader, rank=0, world_size=2)
        from torch.utils.data import DistributedSampler
        assert isinstance(wrapped.sampler, DistributedSampler)
        assert wrapped.batch_size == loader.batch_size

    def test_wrap_loader_preserves_collate_fn(self):
        ddp = DDPStrategy()
        custom_fn = lambda x: x
        loader = DataLoader(TensorDataset(torch.randn(10, 4)), collate_fn=custom_fn)
        wrapped = ddp.wrap_loader(loader, rank=0, world_size=2)
        assert wrapped.collate_fn is custom_fn


# ------------------------------------------------------------------
# FSDPStrategy tests (interface-level)
# ------------------------------------------------------------------

class TestFSDPStrategy:
    def test_is_base_strategy(self):
        fsdp = FSDPStrategy()
        assert isinstance(fsdp, BaseDistributedStrategy)

    def test_default_config(self):
        fsdp = FSDPStrategy()
        assert fsdp._sharding == "full"
        assert fsdp._mp_dtype is None
        assert fsdp._cpu_offload is False
        assert fsdp._act_ckpt is False
        assert fsdp._min_params == 100_000

    def test_custom_config(self):
        fsdp = FSDPStrategy(
            sharding_strategy="shard_grad_op",
            mixed_precision="bf16",
            cpu_offload=True,
            activation_checkpointing=True,
            auto_wrap_min_params=50_000,
        )
        assert fsdp._sharding == "shard_grad_op"
        assert fsdp._mp_dtype == "bf16"
        assert fsdp._cpu_offload is True
        assert fsdp._act_ckpt is True

    def test_wrap_loader(self):
        fsdp = FSDPStrategy()
        loader = _make_loader()
        wrapped = fsdp.wrap_loader(loader, rank=0, world_size=4)
        from torch.utils.data import DistributedSampler
        assert isinstance(wrapped.sampler, DistributedSampler)


# ------------------------------------------------------------------
# Gradient checkpointing tests
# ------------------------------------------------------------------

class TestGradientCheckpointing:
    def test_apply_detects_blocks(self):
        model = SimpleModel(dim=16, n_blocks=4)
        count = apply_gradient_checkpointing(model)
        assert count == 4  # 4 SimpleBlock instances

    def test_apply_explicit_types(self):
        model = SimpleModel(dim=16, n_blocks=4)
        count = apply_gradient_checkpointing(model, block_types={SimpleBlock})
        assert count == 4

    def test_apply_idempotent(self):
        model = SimpleModel(dim=16, n_blocks=3)
        c1 = apply_gradient_checkpointing(model)
        c2 = apply_gradient_checkpointing(model)
        assert c1 == 3
        assert c2 == 0  # already wrapped

    def test_remove_restores_forward(self):
        model = SimpleModel(dim=16, n_blocks=2)
        apply_gradient_checkpointing(model)
        removed = remove_gradient_checkpointing(model)
        assert removed == 2

        # Should work again without checkpointing
        x = torch.randn(2, 16)
        out = model(x)
        assert out.shape == (2, 1)

    def test_forward_produces_same_output(self):
        model = SimpleModel(dim=16, n_blocks=3)
        x = torch.randn(2, 16)

        model.eval()
        ref = model(x).detach().clone()

        apply_gradient_checkpointing(model)
        model.eval()
        ckpt = model(x).detach().clone()

        assert torch.allclose(ref, ckpt, atol=1e-6)

    def test_backward_works_with_checkpointing(self):
        model = SimpleModel(dim=16, n_blocks=3)
        apply_gradient_checkpointing(model)

        x = torch.randn(4, 16)
        out = model(x)
        loss = out.sum()
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_no_checkpointing_in_eval_mode(self):
        """In eval (no_grad), checkpointing should be bypassed."""
        model = SimpleModel(dim=16, n_blocks=2)
        apply_gradient_checkpointing(model)

        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 16)
            out = model(x)
            assert out.shape == (2, 1)

    def test_estimate_memory_savings(self):
        model = SimpleModel(dim=128, n_blocks=8)
        est = estimate_memory_savings(model, seq_length=256, batch_size=1)

        assert "param_memory_mb" in est
        assert "estimated_savings_mb" in est
        assert "estimated_savings_pct" in est
        assert est["estimated_savings_pct"] > 0
        assert est["param_memory_mb"] > 0


# ------------------------------------------------------------------
# Integration: FinetuneStrategy.fit() with checkpointing flag
# ------------------------------------------------------------------

class TestFitWithCheckpointing:
    """Verify that ``gradient_checkpointing=True`` works in the training loop."""

    def _make_model(self):
        """Minimal mock of MolfunStructureModel interface."""
        class MockAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([SimpleBlock(8) for _ in range(2)])
                self.out = nn.Linear(8, 8)

            def forward(self, batch, **kw):
                x = batch if isinstance(batch, torch.Tensor) else batch["x"]
                for b in self.blocks:
                    x = b(x)
                return type("T", (), {"single": self.out(x)})()

            def freeze_trunk(self):
                for p in self.parameters():
                    p.requires_grad = False

        class MockHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 1)

            def forward(self, trunk, **kw):
                return self.fc(trunk.single.mean(dim=-2) if trunk.single.dim() > 2 else trunk.single)

            def loss(self, preds, targets, **kw):
                return {"loss": nn.functional.mse_loss(preds.squeeze(), targets.squeeze())}

        model = type("M", (), {
            "adapter": MockAdapter(),
            "head": MockHead(),
            "device": "cpu",
            "_peft": None,
            "_strategy": None,
            "forward": lambda self, batch, mask=None: {
                "preds": self.head(self.adapter(batch)),
                "trunk_output": self.adapter(batch),
            },
        })()
        model.forward = lambda batch, mask=None: {
            "preds": model.head(model.adapter(batch)),
        }
        return model

    def test_fit_with_checkpointing_flag(self):
        from molfun.training.head_only import HeadOnlyFinetune

        model = self._make_model()

        X = torch.randn(8, 8)
        y = torch.randn(8, 1)
        loader = DataLoader(TensorDataset(X, y), batch_size=4)

        strategy = HeadOnlyFinetune(lr=1e-3, amp=False, loss_fn="mse")
        history = strategy.fit(
            model, loader, epochs=2, verbose=False,
            gradient_checkpointing=True,
        )

        assert len(history) == 2
        assert "train_loss" in history[0]
