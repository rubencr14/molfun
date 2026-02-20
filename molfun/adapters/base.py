"""Base adapter interface that all model adapters must implement."""

from __future__ import annotations
from abc import ABC, abstractmethod

import torch.nn as nn

from molfun.core.types import TrunkOutput


class BaseAdapter(nn.Module, ABC):
    """
    Normalized interface for structure prediction models.

    Every adapter must expose:
    - forward(batch) → TrunkOutput
    - freeze_trunk() / unfreeze_trunk()
    - peft_target_module → nn.Module (the submodule where PEFT is injected)
    - param_summary() → dict
    """

    @abstractmethod
    def forward(self, batch: dict) -> TrunkOutput:
        ...

    @abstractmethod
    def freeze_trunk(self) -> None:
        ...

    @abstractmethod
    def unfreeze_trunk(self) -> None:
        ...

    @property
    @abstractmethod
    def peft_target_module(self) -> nn.Module:
        """Return the submodule where PEFT layers should be injected."""
        ...

    @abstractmethod
    def param_summary(self) -> dict[str, int]:
        ...
