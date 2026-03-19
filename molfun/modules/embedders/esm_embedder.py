"""
ESM-based embedder (ESMFold style).

Uses a pre-trained ESM-2 language model to produce per-residue
representations. The LM is frozen by default — only the projection
layers are trainable.

Optionally extracts pair representations from ESM attention maps
(contact prediction trick from Rao et al. 2021).
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from molfun.modules.embedders.base import BaseEmbedder, EmbedderOutput, EMBEDDER_REGISTRY


@EMBEDDER_REGISTRY.register("esm")
class ESMEmbedder(BaseEmbedder):
    """
    Frozen ESM-2 language model as input embedder.

    Produces single representations from ESM hidden states and
    optionally pair representations from attention maps.

    Args:
        esm_model: ESM model name (e.g. "esm2_t33_650M_UR50D").
        d_single: Output single dimension (projects from ESM hidden dim).
        d_pair: Output pair dimension (0 to disable pair extraction).
        freeze_lm: Whether to freeze the language model weights.
        layer_idx: Which ESM layer to extract representations from (-1 = last).
    """

    def __init__(
        self,
        esm_model: str = "esm2_t33_650M_UR50D",
        d_single: int = 384,
        d_pair: int = 128,
        freeze_lm: bool = True,
        layer_idx: int = -1,
        **kwargs,
    ):
        super().__init__()
        self._d_single = d_single
        self._d_pair = d_pair
        self._esm_model_name = esm_model
        self._layer_idx = layer_idx

        self._esm = None
        self._esm_dim = None
        self._alphabet = None
        self._freeze_lm = freeze_lm

        # Projection layers (initialized lazily after ESM loads)
        self._single_proj = None
        self._pair_proj = None

    def _lazy_init(self, device: torch.device):
        """Load ESM model on first forward pass."""
        if self._esm is not None:
            return

        try:
            import esm as esm_lib
        except ImportError:
            raise ImportError(
                "ESM is required for ESMEmbedder: pip install fair-esm"
            )

        model, alphabet = esm_lib.pretrained.load_model_and_alphabet(
            self._esm_model_name
        )
        self._esm = model.to(device)
        self._alphabet = alphabet
        self._esm_dim = model.embed_dim

        if self._freeze_lm:
            for p in self._esm.parameters():
                p.requires_grad = False

        self._single_proj = nn.Linear(self._esm_dim, self._d_single).to(device)

        if self._d_pair > 0:
            n_heads = model.num_heads if hasattr(model, "num_heads") else 20
            self._pair_proj = nn.Linear(n_heads, self._d_pair).to(device)

    def forward(
        self,
        aatype: torch.Tensor,
        residue_index: torch.Tensor,
        msa: Optional[torch.Tensor] = None,
        msa_mask: Optional[torch.Tensor] = None,
        tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> EmbedderOutput:
        self._lazy_init(aatype.device)

        if tokens is None:
            # aatype (0-20) → ESM tokens. +4 accounts for special tokens offset.
            tokens = aatype + 4

        results = self._esm(
            tokens,
            repr_layers=[self._get_layer_idx()],
            need_head_weights=(self._d_pair > 0),
        )

        hidden = results["representations"][self._get_layer_idx()]
        # Remove BOS/EOS if ESM added them
        if hidden.shape[1] > aatype.shape[1]:
            hidden = hidden[:, 1:aatype.shape[1] + 1, :]

        single = self._single_proj(hidden)

        pair = None
        if self._d_pair > 0 and "attentions" in results:
            # attentions: [B, n_layers, n_heads, L+2, L+2]
            attn = results["attentions"][:, -1, :, 1:-1, 1:-1]  # last layer
            # [B, n_heads, L, L] → [B, L, L, n_heads]
            attn = attn.permute(0, 2, 3, 1)
            # Symmetrize
            attn = (attn + attn.transpose(1, 2)) / 2.0
            pair = self._pair_proj(attn)

        return EmbedderOutput(single=single, pair=pair)

    def _get_layer_idx(self) -> int:
        if self._layer_idx >= 0:
            return self._layer_idx
        n_layers = self._esm.num_layers
        return n_layers + self._layer_idx + 1  # -1 → last layer

    @property
    def d_single(self) -> int:
        return self._d_single

    @property
    def d_pair(self) -> int:
        return self._d_pair
