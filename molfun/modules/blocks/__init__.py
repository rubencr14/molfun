"""
Pluggable transformer blocks for protein structure models.

Each block processes (msa_repr, pair_repr) or (single_repr, pair_repr)
and returns updated representations. Different block architectures
correspond to different model paradigms:

- **EvoformerBlock**: AlphaFold2 — dual-track MSA + pair with triangular ops
- **PairformerBlock**: AlphaFold3/Protenix — single-track + pair with triangular ops
- **SimpleTransformerBlock**: ESMFold-style — single-track only, no pair

Registry
--------
    from molfun.modules.blocks import BLOCK_REGISTRY
    block = BLOCK_REGISTRY.build("evoformer", d_msa=256, d_pair=128)
"""

from molfun.modules.blocks.base import BaseBlock, BLOCK_REGISTRY
from molfun.modules.blocks.evoformer import EvoformerBlock
from molfun.modules.blocks.pairformer import PairformerBlock
from molfun.modules.blocks.simple_transformer import SimpleTransformerBlock

__all__ = [
    "BLOCK_REGISTRY",
    "BaseBlock",
    "EvoformerBlock",
    "PairformerBlock",
    "SimpleTransformerBlock",
]
