"""
Full fine-tuning: all parameters trainable with layer-wise LR decay.

Most expressive, highest risk of overfitting. Use with substantial datasets
(>10k samples) or with strong regularization (EMA, dropout, weight decay).

Layer-wise LR decay assigns lower learning rates to earlier layers:
    lr_layer_i = lr_base * decay_factor ^ (N - i)
This stabilizes training by letting early (general) features change slowly
while later (task-specific) layers adapt faster.
"""

from __future__ import annotations
from typing import Optional

from molfun.training.base import FinetuneStrategy


class FullFinetune(FinetuneStrategy):
    """
    Unfreeze the entire model with layer-wise LR decay.

    Each trunk block gets: ``lr * layer_lr_decay ^ (N - block_index)``.
    The head gets the base LR. Input embedder and structure module
    get separate configurable LRs.

    Works with any adapter — the number of blocks and component access
    is resolved at setup time via the adapter's generic interface.

    Usage::

        strategy = FullFinetune(
            lr=1e-5, lr_head=1e-3,
            layer_lr_decay=0.9,
            warmup_steps=1000, ema_decay=0.999,
            accumulation_steps=8,
        )
        history = strategy.fit(model, train_loader, val_loader, epochs=5)
    """

    def __init__(
        self,
        layer_lr_decay: float = 0.95,
        lr_head: Optional[float] = None,
        lr_embedder: Optional[float] = None,
        lr_structure_module: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_lr_decay = layer_lr_decay
        self.lr_head = lr_head or self.lr * 10
        self._lr_embedder_override = lr_embedder
        self.lr_structure_module = lr_structure_module or self.lr

    def _setup_impl(self, model) -> None:
        model.adapter.unfreeze_trunk()

    def param_groups(self, model) -> list[dict]:
        if model.head is None:
            raise RuntimeError("No head attached to model.")

        blocks = model.adapter.get_trunk_blocks()
        n_blocks = len(blocks)

        lr_embedder = self._lr_embedder_override
        if lr_embedder is None:
            lr_embedder = self.lr * self.layer_lr_decay ** n_blocks

        seen = set()
        groups = []

        embedder = model.adapter.get_input_embedder()
        if embedder is not None:
            params = [p for p in embedder.parameters() if p.requires_grad]
            seen.update(id(p) for p in params)
            if params:
                groups.append({"params": params, "lr": lr_embedder})

        for i, block in enumerate(blocks):
            block_lr = self.lr * (self.layer_lr_decay ** (n_blocks - 1 - i))
            params = [p for p in block.parameters() if p.requires_grad]
            seen.update(id(p) for p in params)
            if params:
                groups.append({"params": params, "lr": block_lr})

        sm = model.adapter.get_structure_module()
        if sm is not None:
            params = [p for p in sm.parameters() if p.requires_grad and id(p) not in seen]
            seen.update(id(p) for p in params)
            if params:
                groups.append({"params": params, "lr": self.lr_structure_module})

        remaining = [
            p for p in model.adapter.parameters()
            if p.requires_grad and id(p) not in seen
        ]
        if remaining:
            groups.append({"params": remaining, "lr": self.lr})

        head_params = list(model.head.parameters())
        groups.append({"params": head_params, "lr": self.lr_head})

        return groups

    def describe(self) -> dict:
        d = super().describe()
        d.update({
            "layer_lr_decay": self.layer_lr_decay,
            "lr_head": self.lr_head,
            "lr_embedder": self._lr_embedder_override,
            "lr_structure_module": self.lr_structure_module,
        })
        return d
