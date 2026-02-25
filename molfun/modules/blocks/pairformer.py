"""
Pairformer block (AlphaFold3 / Protenix style).

Single-track + pair representation (no MSA track). This is the modern
approach used by AlphaFold3 and Protenix: the MSA is pre-processed
into a single representation, then refined jointly with pair features.

Architecture per block:
1. Single self-attention (with pair bias)
2. Single transition
3. Triangular multiplicative update (outgoing)
4. Triangular multiplicative update (incoming)
5. Pair transition
"""

from __future__ import annotations
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from molfun.modules.blocks.base import BaseBlock, BlockOutput, BLOCK_REGISTRY
from molfun.modules.attention.base import ATTENTION_REGISTRY


@BLOCK_REGISTRY.register("pairformer")
class PairformerBlock(BaseBlock):
    """
    Pairformer block with pluggable attention.

    Unlike EvoformerBlock, operates on single [B, L, D] instead of
    MSA [B, N, L, D]. This is the AF3/Protenix paradigm where MSA
    information is compressed into a single representation before the trunk.
    """

    def __init__(
        self,
        d_single: int = 384,
        d_pair: int = 128,
        n_heads: int = 8,
        n_heads_pair: int = 4,
        dropout: float = 0.0,
        attention_cls: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self._d_single = d_single
        self._d_pair = d_pair

        attn_cls = ATTENTION_REGISTRY[attention_cls or "standard"]
        head_dim = d_single // n_heads
        head_dim_pair = d_pair // n_heads_pair
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_heads_pair = n_heads_pair
        self.head_dim_pair = head_dim_pair

        # Single self-attention with pair bias
        self.norm_s = nn.LayerNorm(d_single)
        self.q_proj = nn.Linear(d_single, d_single, bias=False)
        self.k_proj = nn.Linear(d_single, d_single, bias=False)
        self.v_proj = nn.Linear(d_single, d_single, bias=False)
        self.o_proj = nn.Linear(d_single, d_single)
        self.attn = attn_cls(num_heads=n_heads, head_dim=head_dim, dropout=dropout)
        self.pair_bias = nn.Linear(d_pair, n_heads, bias=False)

        # Single transition
        self.norm_ff = nn.LayerNorm(d_single)
        self.ff = nn.Sequential(
            nn.Linear(d_single, d_single * 4),
            nn.SiLU(),
            nn.Linear(d_single * 4, d_single),
        )

        # Triangular updates on pair
        self.tri_out_norm = nn.LayerNorm(d_pair)
        self.tri_out_left = nn.Linear(d_pair, d_pair)
        self.tri_out_right = nn.Linear(d_pair, d_pair)
        self.tri_out_gate_l = nn.Linear(d_pair, d_pair)
        self.tri_out_gate_r = nn.Linear(d_pair, d_pair)
        self.tri_out_proj = nn.Linear(d_pair, d_pair)
        self.tri_out_norm2 = nn.LayerNorm(d_pair)
        self.tri_out_gate = nn.Linear(d_pair, d_pair)

        self.tri_in_norm = nn.LayerNorm(d_pair)
        self.tri_in_left = nn.Linear(d_pair, d_pair)
        self.tri_in_right = nn.Linear(d_pair, d_pair)
        self.tri_in_gate_l = nn.Linear(d_pair, d_pair)
        self.tri_in_gate_r = nn.Linear(d_pair, d_pair)
        self.tri_in_proj = nn.Linear(d_pair, d_pair)
        self.tri_in_norm2 = nn.LayerNorm(d_pair)
        self.tri_in_gate = nn.Linear(d_pair, d_pair)

        # Pair transition
        self.pair_ff_norm = nn.LayerNorm(d_pair)
        self.pair_ff = nn.Sequential(
            nn.Linear(d_pair, d_pair * 4), nn.SiLU(), nn.Linear(d_pair * 4, d_pair),
        )

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        single: torch.Tensor,
        pair: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> BlockOutput:
        assert pair is not None, "PairformerBlock requires pair representation"
        B, L, D = single.shape
        H, Dh = self.n_heads, self.head_dim

        # ── Single self-attention with pair bias ──
        s = self.norm_s(single)
        q = self.q_proj(s).view(B, L, H, Dh).permute(0, 2, 1, 3)
        k = self.k_proj(s).view(B, L, H, Dh).permute(0, 2, 1, 3)
        v = self.v_proj(s).view(B, L, H, Dh).permute(0, 2, 1, 3)
        bias = self.pair_bias(pair).permute(0, 3, 1, 2)  # [B, H, L, L]
        attn_out = self.attn(q, k, v, bias=bias)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, L, D)
        single = single + self.drop(self.o_proj(attn_out))

        # ── Single transition ──
        single = single + self.drop(self.ff(self.norm_ff(single)))

        # ── Triangular update (outgoing) ──
        p = self.tri_out_norm(pair)
        left = self.tri_out_left(p) * torch.sigmoid(self.tri_out_gate_l(p))
        right = self.tri_out_right(p) * torch.sigmoid(self.tri_out_gate_r(p))
        tri = torch.einsum("bikd,bjkd->bijd", left, right)
        tri = self.tri_out_proj(self.tri_out_norm2(tri)) * torch.sigmoid(self.tri_out_gate(p))
        pair = pair + self.drop(tri)

        # ── Triangular update (incoming) ──
        p = self.tri_in_norm(pair)
        left = self.tri_in_left(p) * torch.sigmoid(self.tri_in_gate_l(p))
        right = self.tri_in_right(p) * torch.sigmoid(self.tri_in_gate_r(p))
        tri = torch.einsum("bkid,bkjd->bijd", left, right)
        tri = self.tri_in_proj(self.tri_in_norm2(tri)) * torch.sigmoid(self.tri_in_gate(p))
        pair = pair + self.drop(tri)

        # ── Pair transition ──
        pair = pair + self.drop(self.pair_ff(self.pair_ff_norm(pair)))

        return BlockOutput(single=single, pair=pair)

    @property
    def d_single(self) -> int:
        return self._d_single

    @property
    def d_pair(self) -> int:
        return self._d_pair
