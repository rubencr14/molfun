"""
Training infrastructure helpers.

Utilities shared by all FinetuneStrategy subclasses:
  EMA             Exponential Moving Average of parameters
  build_scheduler LR scheduler with linear warmup + cosine/linear/constant decay
  unpack_batch    Normalize DataLoader output → (features, targets, mask)
  to_device       Recursively move batch tensors to a device
"""

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn


# ======================================================================
# EMA
# ======================================================================

class EMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy:  shadow = decay * shadow + (1 - decay) * param

    Use ``apply()`` to swap EMA weights into the model for evaluation /
    export, then ``restore()`` to return to the live training weights.
    """

    def __init__(self, parameters: list[nn.Parameter], decay: float = 0.999):
        self.decay  = decay
        self.shadow = [p.data.clone() for p in parameters]
        self.backup: list[torch.Tensor] = []
        self._params = parameters

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self._params):
            s.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self):
        """Swap EMA weights into model (saves originals in backup)."""
        self.backup = [p.data.clone() for p in self._params]
        for s, p in zip(self.shadow, self._params):
            p.data.copy_(s)

    def restore(self):
        """Restore original weights from backup."""
        for b, p in zip(self.backup, self._params):
            p.data.copy_(b)
        self.backup = []

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: dict):
        self.shadow = state["shadow"]
        self.decay  = state["decay"]


# ======================================================================
# Scheduler
# ======================================================================

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Build a LambdaLR scheduler with linear warmup followed by decay.

    Args:
        optimizer:     The optimizer to schedule.
        name:          "cosine" | "linear" | "constant"
        warmup_steps:  Steps to linearly ramp up from 0 → 1.
        total_steps:   Total training steps (warmup + decay).
        min_lr:        Minimum LR multiplier at the end of decay (0–1).

    Returns:
        A ``LambdaLR`` scheduler whose multiplier is applied to
        ``optimizer.param_groups[*]["lr"]``.
    """
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if name == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        if name == "cosine":
            return min_lr + (1.0 - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        if name == "linear":
            return min_lr + (1.0 - min_lr) * (1.0 - progress)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ======================================================================
# Batch utilities
# ======================================================================

def unpack_batch(batch_data):
    """
    Normalize a DataLoader batch into ``(features, targets, mask)``.

    Handled formats
    ---------------
    (features_dict, targets)               — affinity, 2-tuple
    (features_dict, targets, mask)         — affinity, 3-tuple
    (features_tensor, targets)             — TensorDataset
    dict with "labels" key                 — affinity dict batch
    dict without "labels" key             — structure fine-tuning (targets = None)

    Returns
    -------
    features : dict | Tensor
    targets  : Tensor | None
    mask     : Tensor | None
    """
    if isinstance(batch_data, (list, tuple)):
        if len(batch_data) == 3:
            return batch_data[0], batch_data[1], batch_data[2]
        features, targets = batch_data[0], batch_data[1]
        mask = features.get("all_atom_mask") if isinstance(features, dict) else None
        return features, targets, mask

    if isinstance(batch_data, dict):
        targets = batch_data.pop("labels") if "labels" in batch_data else None
        mask = batch_data.get("all_atom_mask") or batch_data.get("mask")
        # Avoid tensor.__bool__ by using explicit None checks via .get()
        mask = batch_data.get("all_atom_mask")
        if mask is None:
            mask = batch_data.get("mask")
        return batch_data, targets, mask

    raise ValueError(f"Unsupported batch format: {type(batch_data)}")


def to_device(batch, device):
    """
    Move all tensors in a batch (dict or Tensor) to ``device``.

    Non-tensor values are left untouched.
    """
    if isinstance(batch, dict):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch
