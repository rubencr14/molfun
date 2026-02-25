"""
Evoformer block (AlphaFold2 style).

Dual-track transformer block that jointly processes MSA and pair
representations through:

1. MSA row-wise attention (with pair bias)
2. MSA column-wise attention
3. MSA transition
4. Outer product mean (MSA → pair)
5. Triangular multiplicative update (outgoing)
6. Triangular multiplicative update (incoming)
7. Triangular self-attention (starting node)
8. Triangular self-attention (ending node)
9. Pair transition

This is a research-friendly reimplementation. For production with
pre-trained AF2 weights, use OpenFoldAdapter.
"""

from __future__ import annotations
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from molfun.modules.blocks.base import BaseBlock, BlockOutput, BLOCK_REGISTRY
from molfun.modules.attention.base import BaseAttention, ATTENTION_REGISTRY


@BLOCK_REGISTRY.register("evoformer")
class EvoformerBlock(BaseBlock):
    """
    Evoformer block with pluggable attention.

    The attention mechanism used for MSA row/column attention can be
    swapped by passing ``attention_cls`` (e.g. FlashAttention, GatedAttention).
    """

    def __init__(
        self,
        d_msa: int = 256,
        d_pair: int = 128,
        n_heads_msa: int = 8,
        n_heads_pair: int = 4,
        dropout: float = 0.0,
        attention_cls: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self._d_msa = d_msa
        self._d_pair = d_pair

        attn_cls = ATTENTION_REGISTRY[attention_cls or "standard"]
        head_dim_msa = d_msa // n_heads_msa
        head_dim_pair = d_pair // n_heads_pair

        # MSA row attention with pair bias
        self.msa_row_norm = nn.LayerNorm(d_msa)
        self.msa_row_q = nn.Linear(d_msa, d_msa, bias=False)
        self.msa_row_k = nn.Linear(d_msa, d_msa, bias=False)
        self.msa_row_v = nn.Linear(d_msa, d_msa, bias=False)
        self.msa_row_out = nn.Linear(d_msa, d_msa)
        self.msa_row_attn = attn_cls(num_heads=n_heads_msa, head_dim=head_dim_msa, dropout=dropout)
        self.pair_bias_proj = nn.Linear(d_pair, n_heads_msa, bias=False)
        self.n_heads_msa = n_heads_msa
        self.head_dim_msa = head_dim_msa

        # MSA column attention
        self.msa_col_norm = nn.LayerNorm(d_msa)
        self.msa_col_q = nn.Linear(d_msa, d_msa, bias=False)
        self.msa_col_k = nn.Linear(d_msa, d_msa, bias=False)
        self.msa_col_v = nn.Linear(d_msa, d_msa, bias=False)
        self.msa_col_out = nn.Linear(d_msa, d_msa)
        self.msa_col_attn = attn_cls(num_heads=n_heads_msa, head_dim=head_dim_msa, dropout=dropout)

        # MSA transition
        self.msa_ff_norm = nn.LayerNorm(d_msa)
        self.msa_ff = nn.Sequential(
            nn.Linear(d_msa, d_msa * 4), nn.GELU(), nn.Linear(d_msa * 4, d_msa),
        )

        # Outer product mean: MSA → pair
        self.opm_norm = nn.LayerNorm(d_msa)
        self.opm_left = nn.Linear(d_msa, 32)
        self.opm_right = nn.Linear(d_msa, 32)
        self.opm_out = nn.Linear(32 * 32, d_pair)

        # Pair triangular multiplicative updates (simplified)
        self.tri_mul_out = _TriangularUpdate(d_pair, mode="outgoing")
        self.tri_mul_in = _TriangularUpdate(d_pair, mode="incoming")

        # Pair self-attention (simplified as standard attn over rows/cols)
        self.pair_row_norm = nn.LayerNorm(d_pair)
        self.pair_row_q = nn.Linear(d_pair, d_pair, bias=False)
        self.pair_row_k = nn.Linear(d_pair, d_pair, bias=False)
        self.pair_row_v = nn.Linear(d_pair, d_pair, bias=False)
        self.pair_row_out = nn.Linear(d_pair, d_pair)
        self.pair_row_attn = attn_cls(num_heads=n_heads_pair, head_dim=head_dim_pair, dropout=dropout)
        self.n_heads_pair = n_heads_pair
        self.head_dim_pair = head_dim_pair

        # Pair transition
        self.pair_ff_norm = nn.LayerNorm(d_pair)
        self.pair_ff = nn.Sequential(
            nn.Linear(d_pair, d_pair * 4), nn.GELU(), nn.Linear(d_pair * 4, d_pair),
        )

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        single: torch.Tensor,
        pair: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> BlockOutput:
        msa = single  # [B, N, L, D_msa]
        assert pair is not None, "EvoformerBlock requires pair representation"
        B, N, L, D = msa.shape
        H_m, Dm = self.n_heads_msa, self.head_dim_msa
        H_p, Dp = self.n_heads_pair, self.head_dim_pair

        # ── MSA row attention with pair bias ──
        m = self.msa_row_norm(msa)
        q = self.msa_row_q(m).view(B * N, L, H_m, Dm).permute(0, 2, 1, 3)
        k = self.msa_row_k(m).view(B * N, L, H_m, Dm).permute(0, 2, 1, 3)
        v = self.msa_row_v(m).view(B * N, L, H_m, Dm).permute(0, 2, 1, 3)
        pb = self.pair_bias_proj(pair).permute(0, 3, 1, 2)  # [B, H, L, L]
        pb = pb.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, H_m, L, L)
        attn_out = self.msa_row_attn(q, k, v, bias=pb)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, L, D)
        msa = msa + self.drop(self.msa_row_out(attn_out))

        # ── MSA column attention ──
        m = self.msa_col_norm(msa).permute(0, 2, 1, 3)  # [B, L, N, D]
        m_flat = m.reshape(B * L, N, D)
        q = self.msa_col_q(m_flat).view(B * L, N, H_m, Dm).permute(0, 2, 1, 3)
        k = self.msa_col_k(m_flat).view(B * L, N, H_m, Dm).permute(0, 2, 1, 3)
        v = self.msa_col_v(m_flat).view(B * L, N, H_m, Dm).permute(0, 2, 1, 3)
        attn_out = self.msa_col_attn(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, L, N, D).permute(0, 2, 1, 3)
        msa = msa + self.drop(self.msa_col_out(attn_out))

        # ── MSA transition ──
        msa = msa + self.drop(self.msa_ff(self.msa_ff_norm(msa)))

        # ── Outer product mean ──
        m = self.opm_norm(msa)
        left = self.opm_left(m)   # [B, N, L, 32]
        right = self.opm_right(m) # [B, N, L, 32]
        opm = torch.einsum("bnid,bnjc->bijdc", left, right)  # [B, L, L, 32, 32]
        opm = opm.mean(dim=0) if B == 1 else opm  # average over batch if needed
        opm = opm.reshape(*opm.shape[:3], -1)  # [B, L, L, 32*32]
        pair = pair + self.drop(self.opm_out(opm))

        # ── Triangular updates ──
        pair = pair + self.drop(self.tri_mul_out(pair, pair_mask))
        pair = pair + self.drop(self.tri_mul_in(pair, pair_mask))

        # ── Pair row self-attention ──
        p = self.pair_row_norm(pair)  # [B, L, L, D_p]
        p_flat = p.reshape(B * L, L, self._d_pair)
        q = self.pair_row_q(p_flat).view(B * L, L, H_p, Dp).permute(0, 2, 1, 3)
        k = self.pair_row_k(p_flat).view(B * L, L, H_p, Dp).permute(0, 2, 1, 3)
        v = self.pair_row_v(p_flat).view(B * L, L, H_p, Dp).permute(0, 2, 1, 3)
        attn_out = self.pair_row_attn(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, L, L, self._d_pair)
        pair = pair + self.drop(self.pair_row_out(attn_out))

        # ── Pair transition ──
        pair = pair + self.drop(self.pair_ff(self.pair_ff_norm(pair)))

        return BlockOutput(single=msa, pair=pair)

    @property
    def d_single(self) -> int:
        return self._d_msa

    @property
    def d_pair(self) -> int:
        return self._d_pair


class _TriangularUpdate(nn.Module):
    """Simplified triangular multiplicative update."""

    def __init__(self, d_pair: int, mode: str = "outgoing"):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(d_pair)
        self.left_proj = nn.Linear(d_pair, d_pair)
        self.right_proj = nn.Linear(d_pair, d_pair)
        self.left_gate = nn.Linear(d_pair, d_pair)
        self.right_gate = nn.Linear(d_pair, d_pair)
        self.out_norm = nn.LayerNorm(d_pair)
        self.out_proj = nn.Linear(d_pair, d_pair)
        self.out_gate = nn.Linear(d_pair, d_pair)

    def forward(
        self, pair: torch.Tensor, mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p = self.norm(pair)

        left = self.left_proj(p) * torch.sigmoid(self.left_gate(p))
        right = self.right_proj(p) * torch.sigmoid(self.right_gate(p))

        if mask is not None:
            left = left * mask.unsqueeze(-1)
            right = right * mask.unsqueeze(-1)

        if self.mode == "outgoing":
            out = torch.einsum("bikd,bjkd->bijd", left, right)
        else:
            out = torch.einsum("bkid,bkjd->bijd", left, right)

        out = self.out_proj(self.out_norm(out)) * torch.sigmoid(self.out_gate(p))
        return out
