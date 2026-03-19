"""
OpenFold convenience wrapper.

Thin alias for MolfunStructureModel(name="openfold", ...).
"""

from __future__ import annotations

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
        model: nn.Module | None = None,
        config: object | None = None,
        weights: str | None = None,
        device: str = "cuda",
        head: str | None = None,
        head_config: dict | None = None,
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
