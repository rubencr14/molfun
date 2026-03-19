"""
Pluggable structure prediction modules.

These modules take single/pair representations from the trunk and
predict 3D coordinates. Different approaches:

- **IPA** (Invariant Point Attention): AlphaFold2 default
- **Diffusion**: Denoising diffusion on coordinates (RF-Diffusion style)
- **Equivariant**: SE(3)-equivariant transformer

Registry
--------
    from molfun.modules.structure_module import STRUCTURE_MODULE_REGISTRY
    sm = STRUCTURE_MODULE_REGISTRY.build("ipa", d_single=384, d_pair=128)
"""

from molfun.modules.structure_module.base import (
    BaseStructureModule,
    StructureModuleOutput,
    STRUCTURE_MODULE_REGISTRY,
)
from molfun.modules.structure_module.ipa import IPAStructureModule
from molfun.modules.structure_module.diffusion import DiffusionStructureModule

__all__ = [
    "STRUCTURE_MODULE_REGISTRY",
    "BaseStructureModule",
    "StructureModuleOutput",
    "IPAStructureModule",
    "DiffusionStructureModule",
]
