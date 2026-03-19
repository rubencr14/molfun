"""Tests for ONNX and TorchScript model export."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from molfun.export.torchscript import export_torchscript, _ScriptWrapper


class _DummyModel(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, batch):
        x = batch["aatype"].float()
        return MagicMock(single=self.linear(x))


class _DummyAdapter:
    def __init__(self):
        self.model = _DummyModel()


class _DummyMolfunModel:
    def __init__(self):
        self.adapter = _DummyAdapter()


class TestTorchScriptExport:
    def test_export_trace(self):
        model = _DummyModel(dim=16)

        class SimpleWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, aatype: torch.Tensor, residue_index: torch.Tensor) -> torch.Tensor:
                return self.m.linear(aatype.float())

        wrapper = SimpleWrapper(model)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            scripted = torch.jit.trace(
                wrapper,
                (torch.zeros(1, 16, dtype=torch.long), torch.arange(16).unsqueeze(0)),
            )
            scripted.save(str(path))
            assert path.exists()
            loaded = torch.jit.load(str(path))
            out = loaded(torch.zeros(1, 16, dtype=torch.long), torch.arange(16).unsqueeze(0))
            assert out.shape == (1, 16)

    def test_script_wrapper_dict_output(self):
        class DictModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)

            def forward(self, batch):
                return {"single": self.linear(batch["aatype"].float())}

        wrapper = _ScriptWrapper(DictModel())
        aatype = torch.zeros(1, 8, dtype=torch.long)
        residue_index = torch.arange(8).unsqueeze(0)
        out = wrapper(aatype, residue_index)
        assert out.shape == (1, 8)


class TestEmbeddingCache:
    def test_memory_backend(self):
        from molfun.cache import EmbeddingCache

        cache = EmbeddingCache(backend="memory", max_memory=10)
        assert len(cache) == 0

        key = "MKWVTFISLLLLFSSAYS"
        vec = np.random.randn(256).astype(np.float32)
        cache.put(key, vec)

        hit = cache.get(key)
        assert hit is not None
        np.testing.assert_array_equal(hit, vec)
        assert len(cache) == 1

        assert cache.get("nonexistent") is None

    def test_disk_backend(self):
        from molfun.cache import EmbeddingCache

        with tempfile.TemporaryDirectory() as tmp:
            cache = EmbeddingCache(backend="disk", directory=tmp)

            key = "ACDEFGHIKLMNPQRSTVWY"
            vec = np.random.randn(128).astype(np.float32)
            cache.put(key, vec)

            hit = cache.get(key)
            assert hit is not None
            np.testing.assert_array_almost_equal(hit, vec)

            cache2 = EmbeddingCache(backend="disk", directory=tmp)
            hit2 = cache2.get(key)
            assert hit2 is not None
            np.testing.assert_array_almost_equal(hit2, vec)

    def test_tiered_backend(self):
        from molfun.cache import EmbeddingCache

        with tempfile.TemporaryDirectory() as tmp:
            cache = EmbeddingCache(backend="tiered", directory=tmp, max_memory=5)

            for i in range(8):
                cache.put(f"seq_{i}", np.array([float(i)]))

            assert cache.get("seq_7") is not None
            np.testing.assert_array_almost_equal(cache.get("seq_7"), np.array([7.0]))

    def test_ttl_expiration(self):
        from molfun.cache import EmbeddingCache

        cache = EmbeddingCache(backend="memory", ttl=0.01)
        cache.put("key", np.array([1.0]))

        import time
        time.sleep(0.05)
        assert cache.get("key") is None

    def test_delete_and_clear(self):
        from molfun.cache import EmbeddingCache

        cache = EmbeddingCache(backend="memory", max_memory=100)
        cache.put("a", np.array([1.0]))
        cache.put("b", np.array([2.0]))
        assert len(cache) == 2

        cache.delete("a")
        assert cache.get("a") is None
        assert len(cache) == 1

        cache.clear()
        assert len(cache) == 0

    def test_lru_eviction(self):
        from molfun.cache import EmbeddingCache

        cache = EmbeddingCache(backend="memory", max_memory=3)
        for i in range(5):
            cache.put(f"k{i}", np.array([float(i)]))

        assert cache.get("k0") is None
        assert cache.get("k1") is None
        assert cache.get("k4") is not None

    def test_dict_key(self):
        from molfun.cache import EmbeddingCache

        cache = EmbeddingCache(backend="memory")
        key = {"sequence": "MKWVT", "model": "openfold", "layer": "last"}
        cache.put(key, np.array([42.0]))
        assert cache.get(key) is not None

    def test_stats(self):
        from molfun.cache import EmbeddingCache

        with tempfile.TemporaryDirectory() as tmp:
            cache = EmbeddingCache(backend="tiered", directory=tmp)
            cache.put("x", np.zeros(100, dtype=np.float32))
            stats = cache.stats()
            assert stats["backend"] == "tiered"
            assert "memory" in stats
            assert "disk" in stats

    def test_cached_decorator(self):
        from molfun.cache import EmbeddingCache

        cache = EmbeddingCache(backend="memory")
        call_count = 0

        @cache.cached
        def expensive_embed(seq: str) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return np.array([len(seq)], dtype=np.float32)

        r1 = expensive_embed("ABC")
        r2 = expensive_embed("ABC")
        assert call_count == 1
        np.testing.assert_array_equal(r1, r2)
