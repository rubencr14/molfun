"""Binding affinity prediction head (∆G regression)."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from molfun.core.types import TrunkOutput


class AffinityHead(nn.Module):
    """
    Predicts binding affinity (∆G or pK) from structural representations.
    
    Takes the single representation from TrunkOutput, pools over residues,
    and regresses to a scalar affinity value per sample.
    
    Supports:
    - Mean pooling (default), attention-weighted pooling
    - Optional pair representation fusion
    - MSE or Huber loss
    """

    def __init__(
        self,
        single_dim: int,
        pair_dim: int = 0,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        pool: str = "mean",
        output_dim: int = 1,
    ):
        """
        Args:
            single_dim: Dimension of single representation (C_s from evoformer).
            pair_dim: Dimension of pair representation. If 0, pair repr is not used.
            hidden_dim: MLP hidden dimension.
            num_layers: Number of MLP layers.
            dropout: Dropout rate.
            pool: Pooling strategy ("mean" or "attention").
            output_dim: Output dimension (1 for regression, >1 for classification).
        """
        super().__init__()

        input_dim = single_dim
        if pair_dim > 0:
            self.pair_proj = nn.Linear(pair_dim, single_dim)
        else:
            self.pair_proj = None

        if pool == "attention":
            self.attn_weight = nn.Linear(single_dim, 1)
        else:
            self.attn_weight = None
        self.pool = pool

        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def _pool(self, single: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool over residue dimension: [B, L, D] → [B, D]."""
        if self.pool == "attention" and self.attn_weight is not None:
            # Attention-weighted pooling
            scores = self.attn_weight(single).squeeze(-1)  # [B, L]
            if mask is not None:
                scores = scores.masked_fill(~mask.bool(), float("-inf"))
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
            return (single * weights).sum(dim=-2)  # [B, D]
        else:
            # Mean pooling
            if mask is not None:
                single = single * mask.unsqueeze(-1)
                return single.sum(dim=-2) / mask.sum(dim=-1, keepdim=True).clamp(min=1)
            return single.mean(dim=-2)

    def forward(
        self,
        trunk_output: TrunkOutput,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            trunk_output: Normalized output from model adapter.
            mask: [B, L] residue mask (1 = valid, 0 = padding).
        
        Returns:
            Predictions: [B, output_dim].
        """
        single = trunk_output.single_repr  # [B, L, D_s]

        # Optionally fuse pair representation
        if self.pair_proj is not None and trunk_output.pair_repr is not None:
            pair_pooled = trunk_output.pair_repr.mean(dim=-2)  # [B, L, D_p]
            single = single + self.pair_proj(pair_pooled)

        pooled = self._pool(single, mask)  # [B, D]
        return self.mlp(pooled)  # [B, output_dim]

    def loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: str = "mse",
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            preds: [B, 1] predicted values.
            targets: [B] or [B, 1] target values.
            loss_fn: "mse" or "huber".
        
        Returns:
            Dict with named losses.
        """
        targets = targets.view_as(preds)
        if loss_fn == "huber":
            loss = F.huber_loss(preds, targets)
        else:
            loss = F.mse_loss(preds, targets)
        return {"affinity_loss": loss}
