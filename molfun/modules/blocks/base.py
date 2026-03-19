"""
Abstract base class for trunk blocks.

A block is the repeating unit of the trunk (Evoformer, Pairformer, etc.).
Stacking N blocks forms the core representation-learning part of the model.

The interface supports three paradigms:
- **Dual-track** (AF2 Evoformer): processes both MSA and pair representations
- **Single+pair** (AF3 Pairformer): processes single and pair (no MSA track)
- **Single-only** (ESMFold): processes only single representation
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from molfun.modules.registry import ModuleRegistry

BLOCK_REGISTRY = ModuleRegistry("block")


@dataclass
class BlockOutput:
    """Standardized output from a trunk block."""
    single: Optional[torch.Tensor] = None   # [B, L, D_s] or [B, N, L, D_m]
    pair: Optional[torch.Tensor] = None      # [B, L, L, D_p]


class BaseBlock(ABC, nn.Module):
    """
    A single repeating block of the trunk.

    Subclasses implement the specific architecture (Evoformer, Pairformer,
    simple transformer, etc.) but all accept and return representations
    through the same ``BlockOutput`` interface.
    """

    @abstractmethod
    def forward(
        self,
        single: torch.Tensor,
        pair: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> BlockOutput:
        """
        Process representations through one block.

        Args:
            single: Per-token features.
                    Evoformer: MSA repr [B, N_msa, L, D_msa]
                    Pairformer/Simple: single repr [B, L, D_single]
            pair: Pairwise features [B, L, L, D_pair]. None for single-track models.
            mask: Token mask [B, L] or [B, N, L].
            pair_mask: Pair mask [B, L, L].

        Returns:
            BlockOutput with updated single and pair representations.
        """

    @property
    @abstractmethod
    def d_single(self) -> int:
        """Single/MSA representation dimension."""

    @property
    @abstractmethod
    def d_pair(self) -> int:
        """Pair representation dimension (0 if single-track only)."""
