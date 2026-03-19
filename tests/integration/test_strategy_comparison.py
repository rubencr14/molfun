"""
Integration test: Compare fine-tuning strategies.

Trains the same small model with HeadOnly, LoRA, Partial, and Full,
verifying all produce valid history and that parameter freezing behaves
correctly per strategy.
"""

import pytest

from molfun.training import HeadOnlyFinetune, LoRAFinetune, PartialFinetune, FullFinetune
from tests.integration.conftest import build_custom_model, make_loader


class TestHeadOnly:
    def test_trunk_frozen(self):
        model = build_custom_model()
        strategy = HeadOnlyFinetune(lr=1e-3, amp=False, loss_fn="mse")
        strategy.setup(model)

        trunk_grads = [p.requires_grad for p in model.adapter.parameters()]
        assert not any(trunk_grads), "HeadOnly should freeze all trunk params"
        assert all(p.requires_grad for p in model.head.parameters())

    def test_trains_and_reduces_loss(self):
        model = build_custom_model()
        strategy = HeadOnlyFinetune(lr=1e-2, amp=False, loss_fn="mse")
        h = strategy.fit(model, make_loader(16), epochs=5, verbose=False)
        assert h[-1]["train_loss"] <= h[0]["train_loss"]


class TestLoRA:
    def test_lora_injects_adapters(self):
        model = build_custom_model()
        strategy = LoRAFinetune(rank=4, lr_lora=1e-4, lr_head=1e-3, amp=False, loss_fn="mse")
        strategy.setup(model)

        assert model._peft is not None
        summary = model._peft.summary()
        assert summary.get("lora_params", summary.get("total_params", 0)) > 0

    def test_lora_trains(self):
        model = build_custom_model()
        strategy = LoRAFinetune(rank=4, lr_lora=1e-4, lr_head=1e-3, amp=False, loss_fn="mse")
        h = strategy.fit(model, make_loader(), epochs=2, verbose=False)
        assert len(h) == 2
        assert all(ep["train_loss"] > 0 for ep in h)


class TestPartialFinetune:
    def test_unfreezes_last_blocks(self):
        model = build_custom_model()
        strategy = PartialFinetune(
            unfreeze_last_n=1, lr_trunk=1e-5, lr_head=1e-3,
            amp=False, loss_fn="mse",
        )
        strategy.setup(model)

        blocks = model.adapter.get_trunk_blocks()
        frozen_block = blocks[0]
        unfrozen_block = blocks[-1]

        assert not any(p.requires_grad for p in frozen_block.parameters())
        assert any(p.requires_grad for p in unfrozen_block.parameters())

    def test_trains(self):
        model = build_custom_model()
        strategy = PartialFinetune(
            unfreeze_last_n=1, lr_trunk=1e-4, lr_head=1e-3,
            amp=False, loss_fn="mse",
        )
        h = strategy.fit(model, make_loader(), epochs=2, verbose=False)
        assert len(h) == 2


class TestFullFinetune:
    def test_all_params_trainable(self):
        model = build_custom_model()
        strategy = FullFinetune(lr=1e-4, amp=False, loss_fn="mse")
        strategy.setup(model)

        all_grads = [p.requires_grad for p in model.adapter.parameters()]
        assert all(all_grads), "Full finetune should leave all params trainable"

    def test_trains(self):
        model = build_custom_model()
        strategy = FullFinetune(lr=1e-4, amp=False, loss_fn="mse")
        h = strategy.fit(model, make_loader(), epochs=1, verbose=False)
        assert h[0]["train_loss"] > 0


class TestStrategyComparisonWorkflow:
    """Compare multiple strategies on the same data and build a report."""

    def test_compare_three_strategies(self):
        loader = make_loader(16)
        results: dict[str, float] = {}

        for name, strategy in [
            ("headonly", HeadOnlyFinetune(lr=1e-2, amp=False, loss_fn="mse")),
            ("lora", LoRAFinetune(rank=4, lr_lora=1e-4, lr_head=1e-3, amp=False, loss_fn="mse")),
            ("full", FullFinetune(lr=1e-4, amp=False, loss_fn="mse")),
        ]:
            model = build_custom_model()
            h = strategy.fit(model, loader, epochs=3, verbose=False)
            results[name] = h[-1]["train_loss"]

        assert len(results) == 3
        assert all(v > 0 for v in results.values())
