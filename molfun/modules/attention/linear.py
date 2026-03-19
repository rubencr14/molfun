"""
Linear attention (kernel-based).

Replaces softmax(QK^T)V with φ(Q)·(φ(K)^T·V), reducing complexity from
O(L²) to O(L·D). Useful for very long sequences (whole-genome, long MSAs)
where quadratic attention is the bottleneck.

Reference: Katharopoulos et al., "Transformers are RNNs" (2020).
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from molfun.modules.attention.base import BaseAttention, ATTENTION_REGISTRY


def _elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU+1 feature map: φ(x) = elu(x) + 1."""
    return F.elu(x) + 1.0


@ATTENTION_REGISTRY.register("linear")
class LinearAttention(BaseAttention):
    """
    Linear attention with ELU+1 kernel feature map.

    Computes attention in O(L·D²) instead of O(L²·D), enabling
    sub-quadratic scaling on very long sequences.
    """

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 32,
        dropout: float = 0.0,
        eps: float = 1e-6,
        feature_map: str = "elu",
        **kwargs,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._eps = eps
        self._feature_map = _elu_feature_map

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Apply feature map: [B, H, L, D]
        q_prime = self._feature_map(q)
        k_prime = self._feature_map(k)

        if mask is not None:
            # Zero out masked key positions
            k_prime = k_prime * mask.float().unsqueeze(-1)
            v = v * mask.float().unsqueeze(-1)

        # KV = K^T @ V : [B, H, D, D_v]
        kv = torch.einsum("bhsd,bhsv->bhdv", k_prime, v)

        # Q @ KV : [B, H, Lq, D_v]
        numerator = torch.einsum("bhqd,bhdv->bhqv", q_prime, kv)

        # Normalizer: Q @ sum(K) : [B, H, Lq, 1]
        denominator = torch.einsum(
            "bhqd,bhd->bhq", q_prime, k_prime.sum(dim=-2)
        ).unsqueeze(-1)

        return numerator / (denominator + self._eps)

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim
