"""Tests for attention mechanism implementations."""

import pytest
import torch

from molfun.modules.attention import (
    ATTENTION_REGISTRY,
    StandardAttention,
    FlashAttention,
    LinearAttention,
    GatedAttention,
)


B, H, L, D = 2, 4, 16, 32


@pytest.fixture
def qkv():
    torch.manual_seed(42)
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    return q, k, v


@pytest.fixture
def mask():
    m = torch.ones(B, 1, L, L, dtype=torch.bool)
    m[:, :, :, -2:] = False  # mask out last 2 positions
    return m


@pytest.fixture
def bias():
    return torch.randn(B, H, L, L)


class TestRegistry:
    def test_all_registered(self):
        for name in ("standard", "flash", "linear", "gated"):
            assert name in ATTENTION_REGISTRY

    def test_build(self):
        attn = ATTENTION_REGISTRY.build("standard", num_heads=H, head_dim=D)
        assert isinstance(attn, StandardAttention)
        assert attn.num_heads == H
        assert attn.head_dim == D
        assert attn.embed_dim == H * D


class TestStandardAttention:
    def test_forward_shape(self, qkv):
        q, k, v = qkv
        attn = StandardAttention(num_heads=H, head_dim=D)
        out = attn(q, k, v)
        assert out.shape == (B, H, L, D)

    def test_with_mask(self, qkv, mask):
        q, k, v = qkv
        attn = StandardAttention(num_heads=H, head_dim=D)
        out = attn(q, k, v, mask=mask)
        assert out.shape == (B, H, L, D)

    def test_with_bias(self, qkv, bias):
        q, k, v = qkv
        attn = StandardAttention(num_heads=H, head_dim=D)
        out = attn(q, k, v, bias=bias)
        assert out.shape == (B, H, L, D)

    def test_with_mask_and_bias(self, qkv, mask, bias):
        q, k, v = qkv
        attn = StandardAttention(num_heads=H, head_dim=D)
        out = attn(q, k, v, mask=mask, bias=bias)
        assert out.shape == (B, H, L, D)


class TestFlashAttention:
    def test_forward_shape(self, qkv):
        q, k, v = qkv
        attn = FlashAttention(num_heads=H, head_dim=D)
        out = attn(q, k, v)
        assert out.shape == (B, H, L, D)

    def test_from_standard(self):
        std = StandardAttention(num_heads=H, head_dim=D)
        flash = FlashAttention.from_standard(std)
        assert flash.num_heads == H
        assert flash.head_dim == D

    def test_with_bias(self, qkv, bias):
        q, k, v = qkv
        attn = FlashAttention(num_heads=H, head_dim=D)
        out = attn(q, k, v, bias=bias)
        assert out.shape == (B, H, L, D)


class TestLinearAttention:
    def test_forward_shape(self, qkv):
        q, k, v = qkv
        attn = LinearAttention(num_heads=H, head_dim=D)
        out = attn(q, k, v)
        assert out.shape == (B, H, L, D)

    def test_no_nan(self, qkv):
        q, k, v = qkv
        attn = LinearAttention(num_heads=H, head_dim=D)
        out = attn(q, k, v)
        assert not torch.isnan(out).any()


class TestGatedAttention:
    def test_forward_shape(self, qkv):
        q, k, v = qkv
        attn = GatedAttention(num_heads=H, head_dim=D)
        out = attn(q, k, v)
        assert out.shape == (B, H, L, D)

    def test_has_trainable_gate(self):
        attn = GatedAttention(num_heads=H, head_dim=D)
        gate_params = sum(p.numel() for p in attn.gate_proj.parameters())
        assert gate_params > 0
