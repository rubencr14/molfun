"""
Abstract base class for attention mechanisms.

The contract is intentionally minimal: (q, k, v) in, attended output out.
Bias and mask are optional so the same interface covers MSA row/column
attention, pair attention, and cross-attention variants.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

from molfun.modules.registry import ModuleRegistry

ATTENTION_REGISTRY = ModuleRegistry("attention")


@dataclass
class AttentionConfig:
    """Shared configuration understood by all attention implementations."""
    num_heads: int = 8
    head_dim: int = 32
    dropout: float = 0.0
    bias: bool = True


class BaseAttention(ABC, nn.Module):
    """
    Any attention mechanism that maps (Q, K, V) → output.

    Implementations must at minimum support:
    - Multi-head attention with ``num_heads`` heads of ``head_dim`` each.
    - An optional additive bias tensor (used by Evoformer pair bias).
    - An optional boolean mask (True = attend, False = ignore).

    The input tensors already have the head dimension split out::

        q: [B, H, Lq, D]
        k: [B, H, Lk, D]
        v: [B, H, Lk, D]

    Return shape: ``[B, H, Lq, D]``.
    """

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q: Query  [B, H, Lq, D]
            k: Key    [B, H, Lk, D]
            v: Value  [B, H, Lk, D]
            mask: Boolean mask [B, 1|H, Lq, Lk]. True = attend.
            bias: Additive bias [B, 1|H, Lq, Lk] added to logits.

        Returns:
            Attended output [B, H, Lq, D].
        """

    @property
    @abstractmethod
    def num_heads(self) -> int: ...

    @property
    @abstractmethod
    def head_dim(self) -> int: ...

    @property
    def embed_dim(self) -> int:
        return self.num_heads * self.head_dim

    @classmethod
    def from_config(cls, cfg: AttentionConfig, **overrides) -> "BaseAttention":
        """Build from a dataclass config, with optional field overrides."""
        kwargs = {**cfg.__dict__, **overrides}
        return cls(**kwargs)
