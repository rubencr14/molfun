"""Tests for structure module implementations."""

import pytest
import torch

from molfun.modules.structure_module import (
    STRUCTURE_MODULE_REGISTRY,
    IPAStructureModule,
    DiffusionStructureModule,
    StructureModuleOutput,
)

B, L, D_S, D_P = 2, 16, 64, 32


@pytest.fixture
def inputs():
    torch.manual_seed(42)
    single = torch.randn(B, L, D_S)
    pair = torch.randn(B, L, L, D_P)
    aatype = torch.randint(0, 20, (B, L))
    mask = torch.ones(B, L)
    return single, pair, aatype, mask


class TestRegistry:
    def test_registered(self):
        assert "ipa" in STRUCTURE_MODULE_REGISTRY
        assert "diffusion" in STRUCTURE_MODULE_REGISTRY

    def test_build_ipa(self):
        sm = STRUCTURE_MODULE_REGISTRY.build("ipa", d_single=D_S, d_pair=D_P, n_layers=2)
        assert isinstance(sm, IPAStructureModule)
        assert sm.d_single == D_S
        assert sm.d_pair == D_P


class TestIPAStructureModule:
    def test_forward_shape(self, inputs):
        single, pair, aatype, mask = inputs
        sm = IPAStructureModule(d_single=D_S, d_pair=D_P, n_heads=4, n_layers=2)
        out = sm(single, pair, aatype=aatype, mask=mask)
        assert isinstance(out, StructureModuleOutput)
        assert out.positions.shape == (B, L, 3)

    def test_confidence_produced(self, inputs):
        single, pair, aatype, mask = inputs
        sm = IPAStructureModule(d_single=D_S, d_pair=D_P, n_layers=2)
        out = sm(single, pair)
        assert out.confidence is not None
        assert out.confidence.shape == (B, L)

    def test_single_repr_updated(self, inputs):
        single, pair, aatype, mask = inputs
        sm = IPAStructureModule(d_single=D_S, d_pair=D_P, n_layers=2)
        out = sm(single, pair)
        assert out.single_repr is not None
        assert out.single_repr.shape == (B, L, D_S)

    def test_gradient_flows(self, inputs):
        single, pair, aatype, mask = inputs
        single.requires_grad_(True)
        sm = IPAStructureModule(d_single=D_S, d_pair=D_P, n_layers=2)
        out = sm(single, pair)
        out.positions.sum().backward()
        assert single.grad is not None
        assert single.grad.abs().sum() > 0


class TestDiffusionStructureModule:
    def test_forward_train(self, inputs):
        single, pair, aatype, mask = inputs
        sm = DiffusionStructureModule(
            d_single=D_S, d_pair=D_P, d_model=32, n_layers=2, n_steps=10,
        )
        sm.train()
        out = sm(single, pair)
        assert out.positions.shape == (B, L, 3)
        assert "t" in out.extra

    def test_forward_inference(self, inputs):
        single, pair, aatype, mask = inputs
        sm = DiffusionStructureModule(
            d_single=D_S, d_pair=D_P, d_model=32, n_layers=2, n_steps=5,
        )
        sm.eval()
        out = sm(single, pair)
        assert out.positions.shape == (B, L, 3)
