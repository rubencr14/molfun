"""
molfun.modules — Pluggable components for protein ML research.

This package provides a modular, registry-based system for building
and modifying protein structure prediction models. Every major component
(attention, blocks, structure modules, embedders) has:

1. An **abstract base class** defining the interface contract
2. A **registry** for discovering and instantiating implementations by name
3. **Built-in implementations** covering the main paradigms (AF2, AF3, ESMFold)

Three ways to use this system:

**Registry lookup** — build components by name::

    from molfun.modules.attention import ATTENTION_REGISTRY
    attn = ATTENTION_REGISTRY.build("flash", num_heads=8, head_dim=32)

**Module swapping** — patch components in existing pre-trained models::

    from molfun.modules.swapper import ModuleSwapper
    ModuleSwapper.swap(model, "structure_module", MyCustomSM())

**Model building** — compose custom architectures from components::

    from molfun.modules.builder import ModelBuilder
    model = ModelBuilder(
        embedder="input", block="pairformer", structure_module="ipa",
        n_blocks=24,
    ).build()

Registries
----------
    ATTENTION_REGISTRY          standard, flash, linear, gated
    BLOCK_REGISTRY              evoformer, pairformer, simple_transformer
    STRUCTURE_MODULE_REGISTRY   ipa, diffusion
    EMBEDDER_REGISTRY           input, esm
    PAIR_OP_REGISTRY            (extensible)
"""

from molfun.modules.registry import ModuleRegistry
from molfun.modules.swapper import ModuleSwapper
from molfun.modules.builder import ModelBuilder

from molfun.modules.attention import ATTENTION_REGISTRY
from molfun.modules.blocks import BLOCK_REGISTRY
from molfun.modules.structure_module import STRUCTURE_MODULE_REGISTRY
from molfun.modules.embedders import EMBEDDER_REGISTRY
from molfun.modules.pair_ops import PAIR_OP_REGISTRY

__all__ = [
    "ModuleRegistry",
    "ModuleSwapper",
    "ModelBuilder",
    "ATTENTION_REGISTRY",
    "BLOCK_REGISTRY",
    "STRUCTURE_MODULE_REGISTRY",
    "EMBEDDER_REGISTRY",
    "PAIR_OP_REGISTRY",
]
