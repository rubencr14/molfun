from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class TrunkOutput:
    """Normalized output from any model adapter."""

    single_repr: torch.Tensor  # [B, L, D_s]
    pair_repr: torch.Tensor | None = None  # [B, L, L, D_p]
    structure_coords: torch.Tensor | None = None  # [B, L, 3] or [B, L, 14, 3]
    confidence: torch.Tensor | None = None  # [B, L]
    extra: dict = field(default_factory=dict)


@dataclass
class Batch:
    """Minimal batch container for training/inference."""

    sequences: list[str]
    labels: torch.Tensor | None = None
    metadata: dict | None = None
