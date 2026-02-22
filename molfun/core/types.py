from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class TrunkOutput:
    """Normalized output from any model adapter."""
    single_repr: torch.Tensor                       # [B, L, D_s]
    pair_repr: Optional[torch.Tensor] = None        # [B, L, L, D_p]
    structure_coords: Optional[torch.Tensor] = None # [B, L, 3] or [B, L, 14, 3]
    confidence: Optional[torch.Tensor] = None       # [B, L]
    extra: dict = field(default_factory=dict)


@dataclass
class Batch:
    """Minimal batch container for training/inference."""
    sequences: list[str]
    labels: Optional[torch.Tensor] = None
    metadata: Optional[dict] = None
