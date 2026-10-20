"""Pilot recipes build the strategy they name."""

from pathlib import Path

import pytest

from molfun.families.kinase.pilot import load_pilot_config, make_strategy
from molfun.training import FullFinetune, LoRAFinetune

RECIPES = Path(__file__).resolve().parents[2] / "recipes"


def test_full_recipe_builds_full_finetune():
    strategy = make_strategy(load_pilot_config(RECIPES / "kinase_pilot.yaml")["train"])

    assert isinstance(strategy, FullFinetune)
    assert strategy.amp_dtype == "bfloat16"


def test_lora_recipe_builds_lora_finetune():
    cfg = load_pilot_config(RECIPES / "kinase_pilot_lora.yaml")

    strategy = make_strategy(cfg["train"])

    assert cfg["name"] == "lora"
    assert isinstance(strategy, LoRAFinetune)
    assert strategy.rank == 16
    assert strategy.alpha == 32
    assert strategy.lr_lora == pytest.approx(1e-4)
    assert strategy.amp_dtype == "bfloat16"


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="strategy"):
        make_strategy({"strategy": "partial"})
