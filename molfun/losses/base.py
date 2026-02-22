"""
Base class for all Molfun loss functions + global registry.

Usage
-----
from molfun.losses import LOSS_REGISTRY, LossFunction

# Use a built-in loss by name:
loss_fn = LOSS_REGISTRY["mse"]()
loss    = loss_fn(preds, targets)

# Register a custom loss:
@LOSS_REGISTRY.register("tmscore")
class TMScoreLoss(LossFunction):
    def forward(self, preds, targets=None, batch=None):
        ...
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn


class LossFunction(ABC, nn.Module):
    """
    Abstract base for all Molfun loss functions.

    A LossFunction is a callable nn.Module with a unified signature:

        loss_fn(preds, targets=None, batch=None) -> dict[str, Tensor]

    *preds*   — model predictions (tensor or TrunkOutput, depending on the loss)
    *targets* — ground truth labels (None when GT is embedded in batch)
    *batch*   — raw feature dict forwarded from the DataLoader; required for
                structure losses that compare against atom-coordinate fields.

    Returns a dict mapping loss-term names to scalar tensors so callers can
    log individual terms without knowing loss internals.
    """

    @abstractmethod
    def forward(
        self,
        preds: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        batch: Optional[dict] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss and return a dict of named scalar tensors."""

    # Convenience: allow loss_fn(preds, targets) → dict
    def __call__(self, preds, targets=None, batch=None):
        return super().__call__(preds, targets=targets, batch=batch)


class LossRegistry:
    """
    Simple name → LossFunction class registry.

    Examples
    --------
    LOSS_REGISTRY["mse"]          # returns the class
    LOSS_REGISTRY["mse"]()        # instantiates with defaults
    LOSS_REGISTRY.register("mse") # decorator to add a new entry
    "mse" in LOSS_REGISTRY        # membership test
    list(LOSS_REGISTRY)           # all registered names
    """

    def __init__(self):
        self._registry: dict[str, type[LossFunction]] = {}

    def register(self, name: str):
        """Decorator: ``@LOSS_REGISTRY.register("name")``."""
        def decorator(cls: type[LossFunction]):
            self._registry[name] = cls
            return cls
        return decorator

    def __getitem__(self, name: str) -> type[LossFunction]:
        if name not in self._registry:
            available = sorted(self._registry)
            raise KeyError(
                f"Loss '{name}' not found. Available: {available}"
            )
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __iter__(self):
        return iter(self._registry)

    def __repr__(self) -> str:
        return f"LossRegistry({sorted(self._registry)})"


LOSS_REGISTRY = LossRegistry()
