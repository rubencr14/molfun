"""
FlashAttention wrapper.

Uses PyTorch's ``scaled_dot_product_attention`` which dispatches to
FlashAttention-2 / memory-efficient kernels when available (CUDA, sm80+).
Falls back to the math implementation on CPU or older GPUs.

For research: drop-in replacement for StandardAttention with O(N) memory
and significantly better throughput on long sequences.
"""

from __future__ import annotations
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from molfun.modules.attention.base import BaseAttention, ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("flash")
class FlashAttention(BaseAttention):
    """
    Flash / memory-efficient attention via ``F.scaled_dot_product_attention``.

    Supports the same interface as StandardAttention so it can be
    swapped in anywhere via the registry or ModuleSwapper.
    """

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 32,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._dropout_p = dropout

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # F.scaled_dot_product_attention expects [B, H, L, D]
        # and accepts an additive attn_mask (float) not a boolean mask.
        attn_mask = None
        if bias is not None and mask is not None:
            attn_mask = bias.masked_fill(~mask, float("-inf"))
        elif bias is not None:
            attn_mask = bias
        elif mask is not None:
            attn_mask = torch.zeros_like(q[:, :, :, :1].expand_as(
                torch.empty(q.shape[0], q.shape[1], q.shape[2], k.shape[2],
                            device=q.device)
            ))
            attn_mask = attn_mask.masked_fill(~mask, float("-inf"))

        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self._dropout_p if self.training else 0.0,
            scale=1.0 / math.sqrt(self._head_dim),
        )

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @classmethod
    def from_standard(cls, standard_attn: BaseAttention) -> "FlashAttention":
        """Convert any BaseAttention to FlashAttention, preserving dimensions."""
        return cls(
            num_heads=standard_attn.num_heads,
            head_dim=standard_attn.head_dim,
        )
