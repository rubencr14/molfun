"""
OpenFold convenience wrapper.

Thin alias for MolfunStructureModel(name="openfold", ...).
Kept for backward compatibility.
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
        model = OpenFold(config=cfg, fine_tune=True, peft="lora", head="affinity", ...)
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[object] = None,
        weights: Optional[str] = None,
        device: str = "cuda",
        fine_tune: bool = False,
        peft: Optional[str] = None,
        peft_config: Optional[dict] = None,
        head: Optional[str] = None,
        head_config: Optional[dict] = None,
    ):
        super().__init__(
            name="openfold",
            model=model,
            config=config,
            weights=weights,
            device=device,
            fine_tune=fine_tune,
            peft=peft,
            peft_config=peft_config,
            head=head,
            head_config=head_config,
        )
