"""Tests for ModuleRegistry."""

import pytest
import torch.nn as nn

from molfun.modules.registry import ModuleRegistry


class _DummyBase(nn.Module):
    pass


class _DummyImpl(_DummyBase):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class _NotSubclass(nn.Module):
    pass


class TestModuleRegistry:
    def test_register_and_retrieve(self):
        reg = ModuleRegistry("test")
        reg.register("dummy")(_DummyImpl)
        assert "dummy" in reg
        assert reg["dummy"] is _DummyImpl

    def test_build(self):
        reg = ModuleRegistry("test")
        reg.register("dummy")(_DummyImpl)
        mod = reg.build("dummy", dim=64)
        assert isinstance(mod, _DummyImpl)
        assert mod.dim == 64

    def test_base_class_enforcement(self):
        reg = ModuleRegistry("test", base_class=_DummyBase)
        reg.register("good")(_DummyImpl)

        with pytest.raises(TypeError, match="does not inherit"):
            reg.register("bad")(_NotSubclass)

    def test_duplicate_raises(self):
        reg = ModuleRegistry("test")
        reg.register("dup")(_DummyImpl)
        with pytest.raises(ValueError, match="Duplicate"):
            reg.register("dup")(_DummyImpl)

    def test_missing_key_raises(self):
        reg = ModuleRegistry("test")
        with pytest.raises(KeyError, match="not found"):
            reg["nonexistent"]

    def test_list_and_iter(self):
        reg = ModuleRegistry("test")
        reg.register("b")(_DummyImpl)
        reg.register("a")(_DummyImpl.__class__)  # just to have two entries
        assert isinstance(reg.list(), list)
        names = list(reg)
        assert len(names) == 2

    def test_get_returns_none(self):
        reg = ModuleRegistry("test")
        assert reg.get("missing") is None

    def test_len(self):
        reg = ModuleRegistry("test")
        assert len(reg) == 0
        reg.register("x")(_DummyImpl)
        assert len(reg) == 1

    def test_repr(self):
        reg = ModuleRegistry("test")
        reg.register("x")(_DummyImpl)
        assert "test" in repr(reg)
        assert "x" in repr(reg)
