"""
Gated attention mechanism.

Adds a sigmoid gate on the output of standard attention, inspired by
AlphaFold2's gated self-attention and GLU-style gating in modern
transformer architectures.

    output = sigmoid(gate) * attention(q, k, v)

The gate is a learned linear projection of the query, giving the model
control over how much attention output flows to the next layer.
"""

from __future__ import annotations
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from molfun.modules.attention.base import BaseAttention, ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("gated")
class GatedAttention(BaseAttention):
    """
    Gated multi-head attention: softmax attention with a learned sigmoid gate.

    Used in AlphaFold2's Evoformer (gated self-attention with pair bias).
    """

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 32,
        dropout: float = 0.0,
        gate_init_bias: float = -2.0,
        **kwargs,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._scale = 1.0 / math.sqrt(head_dim)
        self._dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.gate_proj = nn.Linear(head_dim, head_dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_init_bias)

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
        attended = torch.matmul(weights, v)

        gate = torch.sigmoid(self.gate_proj(q))
        return gate * attended

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim
