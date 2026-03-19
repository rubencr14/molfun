"""
Standard scaled dot-product attention.

Reference implementation: straightforward softmax(QK^T / sqrt(d))V
with optional additive bias and mask. Serves as the baseline for
benchmarking alternative attention mechanisms.
"""

from __future__ import annotations
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from molfun.modules.attention.base import BaseAttention, ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("standard")
class StandardAttention(BaseAttention):
    """
    Vanilla scaled dot-product multi-head attention.

    Operates on pre-split heads: inputs are [B, H, L, D].
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
        self._scale = 1.0 / math.sqrt(head_dim)
        self._dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = torch.matmul(q, k.transpose(-2, -1)) * self._scale

        if bias is not None:
            logits = logits + bias
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))

        weights = F.softmax(logits, dim=-1)
        weights = self._dropout(weights)
        return torch.matmul(weights, v)

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim
