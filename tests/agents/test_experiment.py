"""Tests for experiment data structures."""

import json
import pytest
from molfun.agents.experiment import ExperimentConfig, Experiment


class TestExperimentConfig:
    def test_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.embedder == "input"
        assert cfg.block == "pairformer"
        assert cfg.n_blocks == 8
        assert cfg.strategy == "lora"

    def test_round_trip(self):
        cfg = ExperimentConfig(
            block="evoformer",
            block_config={"d_single": 256, "d_pair": 128, "attention_cls": "flash"},
            n_blocks=4,
            strategy="partial",
            name="test-exp",
        )
        d = cfg.to_dict()
        restored = ExperimentConfig.from_dict(d)
        assert restored.block == "evoformer"
        assert restored.n_blocks == 4
        assert restored.block_config["attention_cls"] == "flash"
        assert restored.name == "test-exp"

    def test_short_description(self):
        cfg = ExperimentConfig(block="pairformer", n_blocks=4, strategy="lora",
                               structure_module="ipa")
        desc = cfg.short_description()
        assert "pairformer" in desc
        assert "4" in desc
        assert "lora" in desc


class TestExperiment:
    def test_summary_line(self):
        cfg = ExperimentConfig(name="my-exp")
        exp = Experiment(id="abc123", config=cfg, status="completed",
                         history=[{"train_loss": 0.5, "val_loss": 0.3}],
                         duration_s=120.5)
        line = exp.summary_line()
        assert "abc123" in line
        assert "my-exp" in line
        assert "0.3" in line

    def test_best_val_loss(self):
        exp = Experiment(history=[
            {"train_loss": 1.0, "val_loss": 0.8},
            {"train_loss": 0.5, "val_loss": 0.4},
            {"train_loss": 0.3, "val_loss": 0.5},
        ])
        assert exp.best_val_loss == 0.4

    def test_best_val_loss_empty(self):
        exp = Experiment()
        assert exp.best_val_loss is None

    def test_json_round_trip(self):
        cfg = ExperimentConfig(block="evoformer", n_blocks=6)
        exp = Experiment(id="test1", config=cfg, status="completed",
                         history=[{"val_loss": 0.1}], duration_s=60.0,
                         metrics={"test_mae": 0.05})
        json_str = exp.to_json()
        restored = Experiment.from_json(json_str)
        assert restored.id == "test1"
        assert restored.config.block == "evoformer"
        assert restored.config.n_blocks == 6
        assert restored.status == "completed"
        assert restored.metrics["test_mae"] == 0.05
