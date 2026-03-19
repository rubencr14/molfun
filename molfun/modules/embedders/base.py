"""
Abstract base class for input embedders.

An embedder converts raw inputs (sequence, MSA, residue index) into
initial single and pair representations that feed into the trunk blocks.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from molfun.modules.registry import ModuleRegistry

EMBEDDER_REGISTRY = ModuleRegistry("embedder")


@dataclass
class EmbedderOutput:
    """Standardized output from any embedder."""
    single: torch.Tensor                          # [B, L, D_s] or [B, N, L, D_msa]
    pair: Optional[torch.Tensor] = None           # [B, L, L, D_p]


class BaseEmbedder(ABC, nn.Module):
    """
    Converts raw features → initial representations for trunk blocks.

    Different paradigms:
    - **AF2 InputEmbedder**: aatype + relpos → single; aatype outer → pair; MSA feat → msa
    - **ESM Embedder**: frozen LM → single repr; optional pair from attention maps
    - **Sequence Embedder**: learned embedding table (baseline)
    """

    @abstractmethod
    def forward(
        self,
        aatype: torch.Tensor,
        residue_index: torch.Tensor,
        msa: Optional[torch.Tensor] = None,
        msa_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> EmbedderOutput:
        """
        Args:
            aatype: Residue types [B, L] (int64, 0-20).
            residue_index: Position indices [B, L].
            msa: MSA features [B, N, L, D_msa_feat] (optional).
            msa_mask: MSA mask [B, N, L] (optional).

        Returns:
            EmbedderOutput with initial single and pair representations.
        """

    @property
    @abstractmethod
    def d_single(self) -> int:
        """Output single representation dimension."""

    @property
    @abstractmethod
    def d_pair(self) -> int:
        """Output pair representation dimension (0 if no pair track)."""
