"""
LoRA fine-tuning: frozen trunk + low-rank adaptation + task head.

Inherits from HeadOnlyFinetune because the trunk is still frozen —
we just add trainable LoRA parameters on top.
"""

from __future__ import annotations
from typing import Optional
from molfun.training.head_only import HeadOnlyFinetune
from molfun.peft.lora import MolfunPEFT


class LoRAFinetune(HeadOnlyFinetune):
    """
    Freeze trunk, inject LoRA into attention layers, train LoRA + head.

    Usage:
        strategy = LoRAFinetune(
            rank=8, alpha=16.0,
            target_modules=["linear_q", "linear_v"],
            lr_head=1e-3, lr_lora=1e-4,
            warmup_steps=200, ema_decay=0.999,
            accumulation_steps=4,
        )
        history = strategy.fit(model, train_loader, val_loader, epochs=10)

        # Export merged weights
        model.merge()
    """

    def __init__(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
        use_hf: bool = True,
        lr_head: Optional[float] = None,
        lr_lora: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["linear_q", "linear_v"]
        self.use_hf = use_hf
        self.lr_head = lr_head or self.lr
        self.lr_lora = lr_lora or self.lr

    def _setup_impl(self, model) -> None:
        super()._setup_impl(model)

        peft = MolfunPEFT.lora(
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            target_modules=self.target_modules,
            use_hf=self.use_hf,
        )
        peft.apply(model.adapter.peft_target_module)
        model._peft = peft

    def param_groups(self, model) -> list[dict]:
        if model.head is None:
            raise RuntimeError("No head attached to model.")
        if model._peft is None:
            raise RuntimeError("PEFT not configured — setup() was not called.")
        groups = []
        head_params = list(model.head.parameters())
        if head_params:
            groups.append({"params": head_params, "lr": self.lr_head})
        groups.append(
            {"params": model._peft.trainable_parameters(), "lr": self.lr_lora}
        )
        return groups

    def describe(self) -> dict:
        d = super().describe()
        d.update({
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": self.target_modules,
            "lr_head": self.lr_head,
            "lr_lora": self.lr_lora,
        })
        return d
