"""
Partial fine-tuning: freeze early layers, unfreeze last N evoformer blocks + head.

Middle ground between LoRA (very few params) and full FT (all params).
Good when you have moderate data (~1k-10k samples) and want the trunk
to adapt without the instability of unfreezing everything.
"""

from __future__ import annotations
from typing import Optional
import torch.nn as nn

from molfun.training.base import FinetuneStrategy


class PartialFinetune(FinetuneStrategy):
    """
    Freeze everything except the last `unfreeze_last_n` evoformer blocks,
    the structure module (optionally), and the task head.

    Supports separate LRs for trunk vs head (discriminative fine-tuning).

    Usage:
        strategy = PartialFinetune(
            unfreeze_last_n=6,
            unfreeze_structure_module=True,
            lr_trunk=1e-5, lr_head=1e-3,
            warmup_steps=500, ema_decay=0.999,
        )
        history = strategy.fit(model, train_loader, val_loader, epochs=10)
    """

    def __init__(
        self,
        unfreeze_last_n: int = 4,
        unfreeze_structure_module: bool = False,
        lr_trunk: Optional[float] = None,
        lr_head: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.unfreeze_last_n = unfreeze_last_n
        self.unfreeze_structure_module = unfreeze_structure_module
        self.lr_trunk = lr_trunk or self.lr
        self.lr_head = lr_head or self.lr

    def _setup_impl(self, model) -> None:
        model.adapter.freeze_trunk()

        blocks = model.adapter.get_evoformer_blocks()
        n_blocks = len(blocks)
        start = max(0, n_blocks - self.unfreeze_last_n)
        for block in blocks[start:]:
            for p in block.parameters():
                p.requires_grad = True

        if self.unfreeze_structure_module:
            sm = getattr(model.adapter.model, "structure_module", None)
            if sm is not None:
                for p in sm.parameters():
                    p.requires_grad = True

    def param_groups(self, model) -> list[dict]:
        if model.head is None:
            raise RuntimeError("No head attached to model.")

        trunk_params = [
            p for p in model.adapter.model.parameters() if p.requires_grad
        ]
        head_params = list(model.head.parameters())

        groups = []
        if trunk_params:
            groups.append({"params": trunk_params, "lr": self.lr_trunk})
        groups.append({"params": head_params, "lr": self.lr_head})
        return groups

    def describe(self) -> dict:
        d = super().describe()
        d.update({
            "unfreeze_last_n": self.unfreeze_last_n,
            "unfreeze_structure_module": self.unfreeze_structure_module,
            "lr_trunk": self.lr_trunk,
            "lr_head": self.lr_head,
        })
        return d
