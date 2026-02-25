"""
Experiment data structures for the agent system.

ExperimentConfig describes what to run; Experiment captures the result.
Both are fully JSON-serializable for persistence and LLM consumption.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import time
import uuid


@dataclass
class ExperimentConfig:
    """Fully declarative description of a single experiment."""

    # Architecture
    embedder: str = "input"
    embedder_config: dict = field(default_factory=dict)
    block: str = "pairformer"
    block_config: dict = field(default_factory=dict)
    n_blocks: int = 8
    structure_module: str = "ipa"
    structure_module_config: dict = field(default_factory=dict)

    # Training
    strategy: str = "lora"
    strategy_config: dict = field(default_factory=dict)
    epochs: int = 20

    # Head
    head: str = "affinity"
    head_config: dict = field(default_factory=dict)

    # Metadata
    name: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def short_description(self) -> str:
        parts = [
            f"{self.block}x{self.n_blocks}",
            self.block_config.get("attention_cls", ""),
            self.structure_module,
            self.strategy,
        ]
        return "-".join(p for p in parts if p)


@dataclass
class Experiment:
    """Result of a single experiment run."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    config: ExperimentConfig = field(default_factory=ExperimentConfig)
    history: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[str] = None
    duration_s: float = 0.0
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    checkpoint_path: Optional[str] = None

    @property
    def best_val_loss(self) -> Optional[float]:
        val_losses = [h.get("val_loss") for h in self.history if "val_loss" in h]
        return min(val_losses) if val_losses else None

    @property
    def final_train_loss(self) -> Optional[float]:
        if self.history:
            return self.history[-1].get("train_loss")
        return None

    def summary_line(self) -> str:
        name = self.config.name or self.config.short_description()
        val = f"val={self.best_val_loss:.4f}" if self.best_val_loss is not None else "val=—"
        status_icon = {"completed": "ok", "failed": "FAIL", "running": "..."}
        s = status_icon.get(self.status, "?")
        extra = ""
        if self.metrics:
            parts = [f"{k}={v:.4f}" for k, v in self.metrics.items() if isinstance(v, float)]
            if parts:
                extra = " | " + ", ".join(parts[:3])
        return f"[{self.id}] {s} {name} — {val}{extra} ({self.duration_s:.0f}s)"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Experiment:
        config_data = d.pop("config", {})
        exp = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        exp.config = ExperimentConfig.from_dict(config_data)
        return exp

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, s: str) -> Experiment:
        return cls.from_dict(json.loads(s))
