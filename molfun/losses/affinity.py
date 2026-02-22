"""
Affinity regression / classification losses.

All losses follow the LossFunction interface and are registered in
LOSS_REGISTRY so the training loop can select them by name.

Available losses
----------------
"mse"     MSELoss      — mean squared error (default for ΔG regression)
"mae"     MAELoss      — mean absolute error (more robust to outliers)
"huber"   HuberLoss    — smooth L1 / Huber (good default for noisy labels)
"pearson" PearsonLoss  — 1 − Pearson correlation (optimizes ranking directly)

Usage
-----
from molfun.losses import LOSS_REGISTRY

loss_fn = LOSS_REGISTRY["huber"]()
result  = loss_fn(preds, targets)   # {"huber_loss": tensor}
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F

from molfun.losses.base import LOSS_REGISTRY, LossFunction


@LOSS_REGISTRY.register("mse")
class MSELoss(LossFunction):
    """Mean Squared Error — standard choice for ΔG / pKd regression."""

    def forward(
        self,
        preds: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        batch: Optional[dict] = None,
    ) -> dict[str, torch.Tensor]:
        targets = _check_targets(targets, "MSELoss")
        return {"affinity_loss": F.mse_loss(preds, targets.view_as(preds))}


@LOSS_REGISTRY.register("mae")
class MAELoss(LossFunction):
    """Mean Absolute Error — less sensitive to outliers than MSE."""

    def forward(
        self,
        preds: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        batch: Optional[dict] = None,
    ) -> dict[str, torch.Tensor]:
        targets = _check_targets(targets, "MAELoss")
        return {"affinity_loss": F.l1_loss(preds, targets.view_as(preds))}


@LOSS_REGISTRY.register("huber")
class HuberLoss(LossFunction):
    """
    Huber (smooth L1) loss — behaves like MSE near zero, MAE for large errors.

    Args:
        delta: Threshold between quadratic and linear regions (default 1.0).
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(
        self,
        preds: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        batch: Optional[dict] = None,
    ) -> dict[str, torch.Tensor]:
        targets = _check_targets(targets, "HuberLoss")
        return {
            "affinity_loss": F.huber_loss(
                preds, targets.view_as(preds), delta=self.delta
            )
        }


@LOSS_REGISTRY.register("pearson")
class PearsonLoss(LossFunction):
    """
    1 − Pearson correlation coefficient.

    Optimizes rank ordering directly rather than absolute values.
    Useful when experimental affinities have systematic offsets.
    Requires batch_size ≥ 2.
    """

    def forward(
        self,
        preds: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        batch: Optional[dict] = None,
    ) -> dict[str, torch.Tensor]:
        targets = _check_targets(targets, "PearsonLoss")
        p = preds.view(-1)
        t = targets.view(-1).float()

        if p.numel() < 2:
            # Pearson undefined for a single sample; fall back to MSE
            return {"affinity_loss": F.mse_loss(p, t)}

        p_mean = p.mean()
        t_mean = t.mean()
        cov    = ((p - p_mean) * (t - t_mean)).mean()
        std_p  = p.std(unbiased=False).clamp(min=1e-8)
        std_t  = t.std(unbiased=False).clamp(min=1e-8)
        r      = cov / (std_p * std_t)
        return {"affinity_loss": 1.0 - r}


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _check_targets(targets, name: str) -> torch.Tensor:
    if targets is None:
        raise ValueError(
            f"{name} requires explicit `targets` (affinity labels). "
            "For structure losses (no labels) use OpenFoldLoss instead."
        )
    return targets
