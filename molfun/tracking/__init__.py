"""
molfun.tracking — Experiment tracking for training, agents, and evaluation.

Supports multiple backends (W&B, Comet, MLflow) via a unified interface.
Use CompositeTracker to log to several backends simultaneously.

Quick start::

    from molfun.tracking import ConsoleTracker
    tracker = ConsoleTracker()

    # Or with W&B
    from molfun.tracking import WandbTracker
    tracker = WandbTracker(project="molfun-search")

    # Or multiple at once
    from molfun.tracking import CompositeTracker, ConsoleTracker, WandbTracker
    tracker = CompositeTracker([ConsoleTracker(), WandbTracker(project="molfun")])

    # Use in training
    model.fit(train_loader, val_loader, strategy=strategy, tracker=tracker)

    # Use in agents
    agent = ResearchAgent(llm=llm, tools=tools, tracker=tracker)
"""

from molfun.tracking.base import BaseTracker
from molfun.tracking.console import ConsoleTracker
from molfun.tracking.composite import CompositeTracker

__all__ = [
    "BaseTracker",
    "ConsoleTracker",
    "CompositeTracker",
]


def _lazy_import(name):
    if name == "WandbTracker":
        from molfun.tracking.wandb_tracker import WandbTracker
        return WandbTracker
    elif name == "CometTracker":
        from molfun.tracking.comet_tracker import CometTracker
        return CometTracker
    elif name == "MLflowTracker":
        from molfun.tracking.mlflow_tracker import MLflowTracker
        return MLflowTracker
    elif name == "LangfuseTracker":
        from molfun.tracking.langfuse_tracker import LangfuseTracker
        return LangfuseTracker
    raise AttributeError(f"module 'molfun.tracking' has no attribute {name!r}")


def __getattr__(name):
    return _lazy_import(name)
