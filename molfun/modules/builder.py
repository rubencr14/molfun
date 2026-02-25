"""
Model builder: compose custom models from pluggable components.

ModelBuilder provides a declarative way to assemble a protein structure
prediction model from individual components (embedder, blocks, structure
module), creating a unified nn.Module that conforms to the BaseAdapter
interface so it can be used with MolfunStructureModel.

Usage
-----
    from molfun.modules.builder import ModelBuilder

    # AF3-style Pairformer model
    model = ModelBuilder(
        embedder="input",
        embedder_config={"d_single": 256, "d_pair": 128},
        block="pairformer",
        block_config={"d_single": 256, "d_pair": 128, "attention_cls": "flash"},
        n_blocks=24,
        structure_module="ipa",
        structure_module_config={"d_single": 256, "d_pair": 128},
    ).build()

    # ESMFold-style model
    model = ModelBuilder(
        embedder="esm",
        embedder_config={"esm_model": "esm2_t33_650M_UR50D", "d_single": 384, "d_pair": 0},
        block="simple_transformer",
        block_config={"d_single": 384, "attention_cls": "flash"},
        n_blocks=8,
        structure_module="ipa",
        structure_module_config={"d_single": 384, "d_pair": 128},
    ).build()

    # Use with Molfun's training framework
    molfun_model = MolfunStructureModel.from_custom(model, head="affinity", ...)
"""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from molfun.modules.blocks.base import BLOCK_REGISTRY, BaseBlock, BlockOutput
from molfun.modules.embedders.base import EMBEDDER_REGISTRY, BaseEmbedder
from molfun.modules.structure_module.base import (
    STRUCTURE_MODULE_REGISTRY,
    BaseStructureModule,
    StructureModuleOutput,
)
from molfun.adapters.base import BaseAdapter
from molfun.core.types import TrunkOutput


class ModelBuilder:
    """
    Declarative builder for custom protein models.

    Resolves component names from registries and wires them together
    into a ``BuiltModel`` (which is a ``BaseAdapter``).
    """

    def __init__(
        self,
        embedder: str = "input",
        embedder_config: Optional[dict] = None,
        block: str = "pairformer",
        block_config: Optional[dict] = None,
        n_blocks: int = 8,
        structure_module: str = "ipa",
        structure_module_config: Optional[dict] = None,
    ):
        self.embedder_name = embedder
        self.embedder_config = embedder_config or {}
        self.block_name = block
        self.block_config = block_config or {}
        self.n_blocks = n_blocks
        self.sm_name = structure_module
        self.sm_config = structure_module_config or {}

    def build(self) -> "BuiltModel":
        """Assemble and return the model."""
        embedder = EMBEDDER_REGISTRY.build(self.embedder_name, **self.embedder_config)

        blocks = nn.ModuleList([
            BLOCK_REGISTRY.build(self.block_name, **self.block_config)
            for _ in range(self.n_blocks)
        ])

        sm = STRUCTURE_MODULE_REGISTRY.build(self.sm_name, **self.sm_config)

        return BuiltModel(
            embedder=embedder,
            blocks=blocks,
            structure_module=sm,
            config=self._config_dict(),
        )

    def _config_dict(self) -> dict:
        return {
            "embedder": {"name": self.embedder_name, **self.embedder_config},
            "block": {"name": self.block_name, "n_blocks": self.n_blocks, **self.block_config},
            "structure_module": {"name": self.sm_name, **self.sm_config},
        }


class BuiltModel(BaseAdapter):
    """
    A model assembled from pluggable components.

    Implements ``BaseAdapter`` so it can be wrapped by ``MolfunStructureModel``
    and used with all existing training strategies, heads, and losses.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        blocks: nn.ModuleList,
        structure_module: BaseStructureModule,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.embedder = embedder
        self.blocks = blocks
        self.structure_module = structure_module
        self._config = config or {}

    def forward(self, batch: dict) -> TrunkOutput:
        aatype = batch.get("aatype")
        residue_index = batch.get("residue_index")
        msa = batch.get("msa")
        msa_mask = batch.get("msa_mask")

        # Strip recycling dim if present
        if aatype is not None and aatype.dim() > 2:
            aatype = aatype[..., 0] if aatype.shape[-1] == 1 else aatype
        if residue_index is not None and residue_index.dim() > 2:
            residue_index = residue_index[..., 0] if residue_index.shape[-1] == 1 else residue_index

        emb_out = self.embedder(
            aatype=aatype,
            residue_index=residue_index,
            msa=msa,
            msa_mask=msa_mask,
        )

        single = emb_out.single
        pair = emb_out.pair

        # Blocks may expect 3D [B, L, D] (Pairformer, SimpleTransformer) or
        # 4D [B, N, L, D] (Evoformer). Adapt the embedder output to match.
        block_expects_3d = (
            len(self.blocks) > 0
            and hasattr(self.blocks[0], 'd_pair')
            and not isinstance(self.blocks[0], type(None))
            and single.dim() == 4
        )

        for block in self.blocks:
            # Squeeze MSA dim for blocks that expect [B, L, D]
            inp = single[:, 0] if (single.dim() == 4 and block.d_pair >= 0 and single.shape[1] == 1) else single
            out = block(inp, pair=pair)
            single = out.single
            if out.pair is not None:
                pair = out.pair

        # Extract first row if still MSA-shaped [B, N, L, D]
        if single.dim() == 4:
            single_repr = single[:, 0]
        else:
            single_repr = single

        sm_out = None
        structure_coords = None
        confidence = None

        if pair is not None:
            sm_out = self.structure_module(
                single_repr, pair, aatype=aatype,
            )
            structure_coords = sm_out.positions
            confidence = sm_out.confidence

        return TrunkOutput(
            single_repr=single_repr,
            pair_repr=pair,
            structure_coords=structure_coords,
            confidence=confidence,
            extra={"sm_output": sm_out, "_config": self._config},
        )

    def freeze_trunk(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_trunk(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    @property
    def peft_target_module(self) -> nn.Module:
        return self.blocks

    def get_evoformer_blocks(self) -> nn.ModuleList:
        return self.blocks

    def param_summary(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def summary(self) -> dict:
        return {
            "config": self._config,
            "n_blocks": len(self.blocks),
            "embedder": type(self.embedder).__name__,
            "structure_module": type(self.structure_module).__name__,
            **self.param_summary(),
        }
