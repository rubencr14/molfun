"""
Head-only fine-tuning: freeze the entire trunk, train only the task head.

Fastest to train, least expressive. Good baseline and for very small datasets
where overfitting the trunk is a risk.
"""

from __future__ import annotations
from molfun.training.base import FinetuneStrategy


class HeadOnlyFinetune(FinetuneStrategy):
    """
    Freeze every trunk parameter, train only the prediction head.

    Usage:
        strategy = HeadOnlyFinetune(lr=1e-3, warmup_steps=50)
        history = strategy.fit(model, train_loader, val_loader, epochs=20)
    """

    def _setup_impl(self, model) -> None:
        model.adapter.freeze_trunk()

    def param_groups(self, model) -> list[dict]:
        if model.head is None:
            raise RuntimeError("No head attached to model.")
        return [{"params": list(model.head.parameters()), "lr": self.lr}]
