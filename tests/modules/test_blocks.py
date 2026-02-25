"""Tests for block implementations."""

import pytest
import torch

from molfun.modules.blocks import (
    BLOCK_REGISTRY,
    EvoformerBlock,
    PairformerBlock,
    SimpleTransformerBlock,
    BaseBlock,
)
from molfun.modules.blocks.base import BlockOutput


B, L, N_MSA = 2, 12, 4
D_MSA, D_PAIR = 64, 32
D_SINGLE = 64


class TestRegistry:
    def test_all_registered(self):
        for name in ("evoformer", "pairformer", "simple_transformer"):
            assert name in BLOCK_REGISTRY

    def test_build(self):
        block = BLOCK_REGISTRY.build(
            "pairformer", d_single=D_SINGLE, d_pair=D_PAIR, n_heads=4,
        )
        assert isinstance(block, PairformerBlock)


class TestEvoformerBlock:
    def test_forward_shape(self):
        block = EvoformerBlock(d_msa=D_MSA, d_pair=D_PAIR, n_heads_msa=4, n_heads_pair=2)
        msa = torch.randn(B, N_MSA, L, D_MSA)
        pair = torch.randn(B, L, L, D_PAIR)
        out = block(msa, pair=pair)
        assert isinstance(out, BlockOutput)
        assert out.single.shape == (B, N_MSA, L, D_MSA)
        assert out.pair.shape == (B, L, L, D_PAIR)

    def test_gradient_flows(self):
        block = EvoformerBlock(d_msa=D_MSA, d_pair=D_PAIR, n_heads_msa=4, n_heads_pair=2)
        msa = torch.randn(B, N_MSA, L, D_MSA, requires_grad=True)
        pair = torch.randn(B, L, L, D_PAIR, requires_grad=True)
        out = block(msa, pair=pair)
        loss = out.single.sum() + out.pair.sum()
        loss.backward()
        assert msa.grad is not None
        assert pair.grad is not None

    def test_custom_attention(self):
        block = EvoformerBlock(
            d_msa=D_MSA, d_pair=D_PAIR, n_heads_msa=4, n_heads_pair=2,
            attention_cls="gated",
        )
        msa = torch.randn(B, N_MSA, L, D_MSA)
        pair = torch.randn(B, L, L, D_PAIR)
        out = block(msa, pair=pair)
        assert out.single.shape == (B, N_MSA, L, D_MSA)


class TestPairformerBlock:
    def test_forward_shape(self):
        block = PairformerBlock(d_single=D_SINGLE, d_pair=D_PAIR, n_heads=4, n_heads_pair=2)
        single = torch.randn(B, L, D_SINGLE)
        pair = torch.randn(B, L, L, D_PAIR)
        out = block(single, pair=pair)
        assert out.single.shape == (B, L, D_SINGLE)
        assert out.pair.shape == (B, L, L, D_PAIR)

    def test_gradient_flows(self):
        block = PairformerBlock(d_single=D_SINGLE, d_pair=D_PAIR, n_heads=4, n_heads_pair=2)
        single = torch.randn(B, L, D_SINGLE, requires_grad=True)
        pair = torch.randn(B, L, L, D_PAIR, requires_grad=True)
        out = block(single, pair=pair)
        (out.single.sum() + out.pair.sum()).backward()
        assert single.grad is not None
        assert pair.grad is not None


class TestSimpleTransformerBlock:
    def test_forward_shape(self):
        block = SimpleTransformerBlock(d_single=D_SINGLE, n_heads=4)
        single = torch.randn(B, L, D_SINGLE)
        out = block(single)
        assert out.single.shape == (B, L, D_SINGLE)
        assert out.pair is None

    def test_d_pair_is_zero(self):
        block = SimpleTransformerBlock(d_single=D_SINGLE, n_heads=4)
        assert block.d_pair == 0

    def test_passthrough_pair(self):
        block = SimpleTransformerBlock(d_single=D_SINGLE, n_heads=4)
        single = torch.randn(B, L, D_SINGLE)
        pair = torch.randn(B, L, L, D_PAIR)
        out = block(single, pair=pair)
        assert out.pair is pair  # should be passed through unchanged

    def test_custom_attention(self):
        block = SimpleTransformerBlock(
            d_single=D_SINGLE, n_heads=4, attention_cls="flash",
        )
        single = torch.randn(B, L, D_SINGLE)
        out = block(single)
        assert out.single.shape == (B, L, D_SINGLE)
