"""
Simple transformer block (ESMFold style).

Single-track only: no pair representation, no triangular operations.
Suitable for language-model-based structure prediction (ESMFold, ESM-IF)
where the trunk is a standard transformer over residue tokens.

Architecture per block:
1. Self-attention (pre-norm)
2. Feed-forward (pre-norm, SwiGLU)
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from molfun.modules.blocks.base import BaseBlock, BlockOutput, BLOCK_REGISTRY
from molfun.modules.attention.base import ATTENTION_REGISTRY


@BLOCK_REGISTRY.register("simple_transformer")
class SimpleTransformerBlock(BaseBlock):
    """
    Single-track transformer block with pluggable attention.

    No pair representation — designed for ESMFold-style models
    or as a lightweight baseline for ablation studies.
    """

    def __init__(
        self,
        d_single: int = 1280,
        n_heads: int = 20,
        ff_factor: int = 4,
        dropout: float = 0.0,
        attention_cls: Optional[str] = None,
        use_swiglu: bool = True,
        **kwargs,
    ):
        super().__init__()
        self._d_single = d_single
        head_dim = d_single // n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim

        attn_cls = ATTENTION_REGISTRY[attention_cls or "standard"]

        self.norm1 = nn.LayerNorm(d_single)
        self.q_proj = nn.Linear(d_single, d_single, bias=False)
        self.k_proj = nn.Linear(d_single, d_single, bias=False)
        self.v_proj = nn.Linear(d_single, d_single, bias=False)
        self.o_proj = nn.Linear(d_single, d_single)
        self.attn = attn_cls(num_heads=n_heads, head_dim=head_dim, dropout=dropout)

        self.norm2 = nn.LayerNorm(d_single)
        d_ff = d_single * ff_factor

        if use_swiglu:
            self.ff = _SwiGLU(d_single, d_ff, dropout)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_single, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_single),
                nn.Dropout(dropout),
            )

    def forward(
        self,
        single: torch.Tensor,
        pair: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> BlockOutput:
        B, L, D = single.shape
        H, Dh = self.n_heads, self.head_dim

        # Self-attention
        s = self.norm1(single)
        q = self.q_proj(s).view(B, L, H, Dh).permute(0, 2, 1, 3)
        k = self.k_proj(s).view(B, L, H, Dh).permute(0, 2, 1, 3)
        v = self.v_proj(s).view(B, L, H, Dh).permute(0, 2, 1, 3)

        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2).bool()  # [B, 1, 1, L]

        attn_out = self.attn(q, k, v, mask=attn_mask)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, L, D)
        single = single + self.o_proj(attn_out)

        # Feed-forward
        single = single + self.ff(self.norm2(single))

        return BlockOutput(single=single, pair=pair)

    @property
    def d_single(self) -> int:
        return self._d_single

    @property
    def d_pair(self) -> int:
        return 0


class _SwiGLU(nn.Module):
    """SwiGLU feed-forward: gate(x) * linear(x), used in LLaMA/ESM2."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
