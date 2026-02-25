"""Tests for ModuleSwapper."""

import pytest
import torch
import torch.nn as nn

from molfun.modules.swapper import ModuleSwapper


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = nn.Linear(10, 32)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(32, 32), nn.ReLU()) for _ in range(3)
        ])
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        x = self.embedder(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class _Replacement(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32)
        self._is_replacement = True

    def forward(self, x):
        return self.linear(x)


class TestSwapSingle:
    def test_swap_named_module(self):
        model = _SimpleModel()
        old_head = model.head
        new_head = nn.Linear(32, 2)
        returned = ModuleSwapper.swap(model, "head", new_head)
        assert returned is old_head
        assert model.head is new_head
        assert model.head.out_features == 2

    def test_swap_nested_module(self):
        model = _SimpleModel()
        replacement = _Replacement()
        ModuleSwapper.swap(model, "blocks.1", replacement)
        assert hasattr(model.blocks[1], '_is_replacement')

    def test_swap_nonexistent_raises(self):
        model = _SimpleModel()
        with pytest.raises(KeyError):
            ModuleSwapper.swap(model, "nonexistent", nn.Linear(1, 1))


class TestSwapAll:
    def test_swap_by_pattern(self):
        model = _SimpleModel()
        count = ModuleSwapper.swap_all(
            model, r"blocks\.\d+$",
            factory=lambda name, old: _Replacement(),
        )
        assert count == 3
        for i in range(3):
            assert hasattr(model.blocks[i], '_is_replacement')

    def test_swap_by_type(self):
        model = _SimpleModel()
        count = ModuleSwapper.swap_by_type(
            model, nn.ReLU,
            factory=lambda name, old: nn.GELU(),
        )
        assert count == 3

    def test_swap_zero_matches(self):
        model = _SimpleModel()
        count = ModuleSwapper.swap_all(
            model, "nonexistent_pattern",
            factory=lambda name, old: nn.Identity(),
        )
        assert count == 0


class TestDiscover:
    def test_discover_all(self):
        model = _SimpleModel()
        modules = ModuleSwapper.discover(model)
        names = [n for n, _ in modules]
        assert "embedder" in names
        assert "head" in names

    def test_discover_with_pattern(self):
        model = _SimpleModel()
        modules = ModuleSwapper.discover(model, pattern="blocks")
        assert len(modules) > 0
        assert all("blocks" in name for name, _ in modules)

    def test_discover_by_type(self):
        model = _SimpleModel()
        modules = ModuleSwapper.discover(model, module_type=nn.Linear)
        assert len(modules) >= 2  # at least embedder and head

    def test_summary(self):
        model = _SimpleModel()
        s = ModuleSwapper.summary(model)
        assert "embedder" in s
        assert "head" in s


class TestTransferWeights:
    def test_transfer_matching_shapes(self):
        model = _SimpleModel()
        old_embedder = model.embedder
        old_weight = old_embedder.weight.clone()
        new_embedder = nn.Linear(10, 32)

        ModuleSwapper.swap(model, "embedder", new_embedder, transfer_weights=True)
        assert torch.allclose(model.embedder.weight, old_weight)
