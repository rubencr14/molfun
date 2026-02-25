"""
Abstract base class for structure prediction modules.

A structure module sits after the trunk (Evoformer / Pairformer / ESM)
and converts learned representations into 3D atomic coordinates.

The output is a standardized ``StructureModuleOutput`` dataclass so
that downstream heads, losses, and analysis tools work regardless of
which structure module variant is used.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from molfun.modules.registry import ModuleRegistry

STRUCTURE_MODULE_REGISTRY = ModuleRegistry("structure_module")


@dataclass
class StructureModuleOutput:
    """Standardized output from any structure prediction module."""
    positions: torch.Tensor                          # [B, L, 3] Cα or [B, L, n_atoms, 3]
    frames: Optional[torch.Tensor] = None            # [B, L, 4, 4] backbone rigid frames
    confidence: Optional[torch.Tensor] = None        # [B, L] per-residue confidence (pLDDT-like)
    single_repr: Optional[torch.Tensor] = None       # [B, L, D] updated single repr
    extra: dict = field(default_factory=dict)


class BaseStructureModule(ABC, nn.Module):
    """
    Maps (single_repr, pair_repr) → 3D structure.

    Different paradigms:
    - **IPA** (AF2): iterative refinement with invariant point attention
    - **Diffusion** (RF-Diffusion/AF3): denoising diffusion on frames
    - **Equivariant** (SE3-Transformers): equivariant message passing

    All must produce a ``StructureModuleOutput`` with at minimum
    the ``positions`` field populated.
    """

    @abstractmethod
    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        aatype: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> StructureModuleOutput:
        """
        Predict 3D coordinates from representations.

        Args:
            single: Per-residue features [B, L, D_single].
            pair:   Pairwise features    [B, L, L, D_pair].
            aatype: Residue types        [B, L] (int64, 0-20).
            mask:   Residue mask         [B, L] (1 = valid).

        Returns:
            StructureModuleOutput with predicted coordinates.
        """

    @property
    @abstractmethod
    def d_single(self) -> int:
        """Expected single representation dimension."""

    @property
    @abstractmethod
    def d_pair(self) -> int:
        """Expected pair representation dimension."""
