"""
OpenFold convenience wrapper.

Thin alias for MolfunStructureModel(name="openfold", ...).
"""

from __future__ import annotations
from typing import Optional
import torch.nn as nn

from molfun.models.structure import MolfunStructureModel


class OpenFold(MolfunStructureModel):
    """
    OpenFold shortcut — equivalent to MolfunStructureModel("openfold", ...).

    Usage:
        model = OpenFold(config=cfg, weights="ckpt.pt")
        model = OpenFold(config=cfg, head="affinity", head_config={...})
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[object] = None,
        weights: Optional[str] = None,
        device: str = "cuda",
        head: Optional[str] = None,
        head_config: Optional[dict] = None,
    ):
        super().__init__(
            name="openfold",
            model=model,
            config=config,
            weights=weights,
            device=device,
            head=head,
            head_config=head_config,
        )
