"""
AF2-style input embedder.

Produces initial single, pair, and MSA representations from:
- amino acid type (one-hot)
- relative position encoding
- MSA features (if available)

This is a self-contained reimplementation for custom model building.
For production with pre-trained AF2 weights, use OpenFoldAdapter.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from molfun.modules.embedders.base import BaseEmbedder, EmbedderOutput, EMBEDDER_REGISTRY


@EMBEDDER_REGISTRY.register("input")
class InputEmbedder(BaseEmbedder):
    """
    AlphaFold2-style input embedding.

    Creates:
    - single repr from aatype one-hot
    - pair repr from outer product of aatype + relative position
    - MSA repr from MSA features (or broadcasts single if no MSA)
    """

    def __init__(
        self,
        d_single: int = 256,
        d_pair: int = 128,
        d_msa: int = 256,
        n_aa_types: int = 22,
        max_relpos: int = 32,
        **kwargs,
    ):
        super().__init__()
        self._d_single = d_single
        self._d_pair = d_pair
        self._d_msa = d_msa
        self._max_relpos = max_relpos

        self.aa_embed = nn.Embedding(n_aa_types, d_single)

        # Pair: left aa + right aa + relative position
        self.pair_left = nn.Linear(d_single, d_pair)
        self.pair_right = nn.Linear(d_single, d_pair)
        n_relpos = 2 * max_relpos + 1
        self.relpos_embed = nn.Embedding(n_relpos, d_pair)

        # MSA
        self.msa_proj = nn.Linear(d_single, d_msa)

    def forward(
        self,
        aatype: torch.Tensor,
        residue_index: torch.Tensor,
        msa: Optional[torch.Tensor] = None,
        msa_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> EmbedderOutput:
        B, L = aatype.shape

        # Single representation
        single = self.aa_embed(aatype)  # [B, L, D_s]

        # Pair representation
        left = self.pair_left(single)    # [B, L, D_p]
        right = self.pair_right(single)  # [B, L, D_p]
        pair = left.unsqueeze(2) + right.unsqueeze(1)  # [B, L, L, D_p]

        # Relative position encoding
        rel_pos = residue_index.unsqueeze(2) - residue_index.unsqueeze(1)
        rel_pos = rel_pos.clamp(-self._max_relpos, self._max_relpos) + self._max_relpos
        pair = pair + self.relpos_embed(rel_pos)

        # MSA representation
        msa_repr = self.msa_proj(single).unsqueeze(1)  # [B, 1, L, D_msa]

        return EmbedderOutput(single=msa_repr, pair=pair)

    @property
    def d_single(self) -> int:
        return self._d_single

    @property
    def d_pair(self) -> int:
        return self._d_pair
