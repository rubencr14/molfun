"""Tests for ModelBuilder and BuiltModel."""

import pytest
import torch

from molfun.modules.builder import ModelBuilder, BuiltModel
from molfun.adapters.base import BaseAdapter
from molfun.core.types import TrunkOutput


B, L = 2, 16


@pytest.fixture
def batch():
    return {
        "aatype": torch.randint(0, 20, (B, L)),
        "residue_index": torch.arange(L).unsqueeze(0).expand(B, -1),
    }


class TestModelBuilder:
    def test_build_pairformer(self):
        model = ModelBuilder(
            embedder="input",
            embedder_config={"d_single": 64, "d_pair": 32, "d_msa": 64},
            block="pairformer",
            block_config={"d_single": 64, "d_pair": 32, "n_heads": 4, "n_heads_pair": 2},
            n_blocks=2,
            structure_module="ipa",
            structure_module_config={"d_single": 64, "d_pair": 32, "n_heads": 4, "n_layers": 2},
        ).build()
        assert isinstance(model, BuiltModel)
        assert isinstance(model, BaseAdapter)
        assert len(model.blocks) == 2

    def test_build_simple_transformer(self):
        model = ModelBuilder(
            embedder="input",
            embedder_config={"d_single": 64, "d_pair": 32, "d_msa": 64},
            block="simple_transformer",
            block_config={"d_single": 64, "n_heads": 4},
            n_blocks=2,
            structure_module="ipa",
            structure_module_config={"d_single": 64, "d_pair": 32, "n_layers": 2},
        ).build()
        assert isinstance(model, BuiltModel)

    def test_forward(self, batch):
        model = ModelBuilder(
            embedder="input",
            embedder_config={"d_single": 64, "d_pair": 32, "d_msa": 64},
            block="pairformer",
            block_config={"d_single": 64, "d_pair": 32, "n_heads": 4, "n_heads_pair": 2},
            n_blocks=2,
            structure_module="ipa",
            structure_module_config={"d_single": 64, "d_pair": 32, "n_heads": 4, "n_layers": 2},
        ).build()
        out = model(batch)
        assert isinstance(out, TrunkOutput)
        assert out.single_repr.shape == (B, L, 64)
        assert out.pair_repr.shape == (B, L, L, 32)
        assert out.structure_coords.shape == (B, L, 3)

    def test_gradient_flows(self, batch):
        model = ModelBuilder(
            embedder="input",
            embedder_config={"d_single": 64, "d_pair": 32, "d_msa": 64},
            block="pairformer",
            block_config={"d_single": 64, "d_pair": 32, "n_heads": 4, "n_heads_pair": 2},
            n_blocks=2,
            structure_module="ipa",
            structure_module_config={"d_single": 64, "d_pair": 32, "n_heads": 4, "n_layers": 2},
        ).build()
        out = model(batch)
        loss = out.single_repr.sum() + out.structure_coords.sum()
        loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert len(trainable) > 0

    def test_freeze_unfreeze(self):
        model = ModelBuilder(
            embedder="input",
            embedder_config={"d_single": 64, "d_pair": 32, "d_msa": 64},
            block="pairformer",
            block_config={"d_single": 64, "d_pair": 32, "n_heads": 4, "n_heads_pair": 2},
            n_blocks=2,
            structure_module="ipa",
            structure_module_config={"d_single": 64, "d_pair": 32, "n_layers": 2},
        ).build()

        model.freeze_trunk()
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert trainable == 0

        model.unfreeze_trunk()
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert trainable > 0

    def test_param_summary(self):
        model = ModelBuilder(
            embedder="input",
            embedder_config={"d_single": 64, "d_pair": 32, "d_msa": 64},
            block="pairformer",
            block_config={"d_single": 64, "d_pair": 32, "n_heads": 4, "n_heads_pair": 2},
            n_blocks=2,
            structure_module="ipa",
            structure_module_config={"d_single": 64, "d_pair": 32, "n_layers": 2},
        ).build()

        summary = model.param_summary()
        assert summary["total"] > 0
        assert summary["total"] == summary["trainable"] + summary["frozen"]

    def test_peft_target_module(self):
        model = ModelBuilder(
            embedder="input",
            embedder_config={"d_single": 64, "d_pair": 32, "d_msa": 64},
            block="pairformer",
            block_config={"d_single": 64, "d_pair": 32, "n_heads": 4, "n_heads_pair": 2},
            n_blocks=2,
            structure_module="ipa",
            structure_module_config={"d_single": 64, "d_pair": 32, "n_layers": 2},
        ).build()
        assert model.peft_target_module is model.blocks
