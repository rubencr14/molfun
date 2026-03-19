"""
Molfun tools exposed to LLM agents via function/tool calling.

Each tool wraps a Molfun API call, accepts JSON-serializable arguments,
and returns a string result the LLM can reason about.
"""

from __future__ import annotations
from typing import Optional
import time
import json
import traceback

import torch
from torch.utils.data import DataLoader

from molfun.agents.experiment import ExperimentConfig, Experiment

STRATEGY_REGISTRY = {}


def _ensure_strategy_registry():
    if STRATEGY_REGISTRY:
        return
    from molfun.training import (
        HeadOnlyFinetune, LoRAFinetune, PartialFinetune, FullFinetune,
    )
    STRATEGY_REGISTRY.update({
        "head_only": HeadOnlyFinetune,
        "lora": LoRAFinetune,
        "partial": PartialFinetune,
        "full": FullFinetune,
    })


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_components",
            "description": (
                "List all available model components (attention mechanisms, "
                "block types, structure modules, embedders) and training "
                "strategies. Call this first to understand what you can build."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_and_train",
            "description": (
                "Build a custom model from components and train it. "
                "Returns experiment ID and training history with metrics. "
                "This is the main action — each call runs a full experiment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short name for this experiment (e.g. 'pairformer-flash-lora8')",
                    },
                    "embedder": {
                        "type": "string",
                        "description": "Embedder type: 'input' (AF2-style) or 'esm' (ESM-2)",
                        "default": "input",
                    },
                    "embedder_config": {
                        "type": "object",
                        "description": "Embedder kwargs (d_single, d_pair, d_msa, ...)",
                        "default": {},
                    },
                    "block": {
                        "type": "string",
                        "description": "Block type: 'pairformer', 'evoformer', or 'simple_transformer'",
                    },
                    "block_config": {
                        "type": "object",
                        "description": "Block kwargs (d_single, d_pair, n_heads, attention_cls, ...)",
                        "default": {},
                    },
                    "n_blocks": {
                        "type": "integer",
                        "description": "Number of trunk blocks",
                        "default": 8,
                    },
                    "structure_module": {
                        "type": "string",
                        "description": "Structure module: 'ipa' or 'diffusion'",
                        "default": "ipa",
                    },
                    "structure_module_config": {
                        "type": "object",
                        "description": "Structure module kwargs (d_single, d_pair, n_layers, ...)",
                        "default": {},
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Training strategy: 'head_only', 'lora', 'partial', 'full'",
                        "default": "lora",
                    },
                    "strategy_config": {
                        "type": "object",
                        "description": "Strategy kwargs (rank, lr_lora, lr_head, warmup_steps, ema_decay, ...)",
                        "default": {},
                    },
                    "head": {
                        "type": "string",
                        "description": "Head type: 'affinity' or 'structure'",
                        "default": "affinity",
                    },
                    "head_config": {
                        "type": "object",
                        "description": "Head kwargs (single_dim, hidden_dim, ...)",
                        "default": {},
                    },
                    "epochs": {
                        "type": "integer",
                        "description": "Number of training epochs",
                        "default": 10,
                    },
                },
                "required": ["name", "block", "block_config"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_experiment",
            "description": "Get full details of a specific experiment by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                },
                "required": ["experiment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_journal",
            "description": (
                "Get the experiment journal: summary of all experiments run so "
                "far, including best result, top experiments, and recent runs."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_best_model",
            "description": "Save the best model checkpoint to disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to save to"},
                    "experiment_id": {
                        "type": "string",
                        "description": "Experiment ID. If omitted, saves the overall best.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "Signal that you are finished. Call this when you've found a "
                "satisfactory result or exhausted the budget. Include a summary "
                "of findings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Final summary of findings and best configuration.",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]


class MolfunTools:
    """
    Executable tools that connect an LLM agent to Molfun's APIs.

    Initialize with data loaders and device, then call ``execute()``
    for each tool call the LLM makes.
    """

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self._models: dict[str, object] = {}  # experiment_id -> MolfunStructureModel
        self._done = False
        self._done_summary: Optional[str] = None

    @property
    def schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def done_summary(self) -> Optional[str]:
        return self._done_summary

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return a string result for the LLM."""
        handlers = {
            "list_components": self._list_components,
            "build_and_train": self._build_and_train,
            "get_experiment": self._get_experiment,
            "get_journal": self._get_journal,
            "save_best_model": self._save_best_model,
            "done": self._done_handler,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return f"Error: Unknown tool '{tool_name}'. Available: {list(handlers)}"
        try:
            return handler(**arguments)
        except Exception as e:
            return f"Error executing {tool_name}: {type(e).__name__}: {e}\n{traceback.format_exc()}"

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _list_components(self) -> str:
        from molfun.modules import (
            ATTENTION_REGISTRY, BLOCK_REGISTRY,
            STRUCTURE_MODULE_REGISTRY, EMBEDDER_REGISTRY,
        )
        _ensure_strategy_registry()

        return json.dumps({
            "attention": ATTENTION_REGISTRY.list(),
            "blocks": BLOCK_REGISTRY.list(),
            "structure_modules": STRUCTURE_MODULE_REGISTRY.list(),
            "embedders": EMBEDDER_REGISTRY.list(),
            "strategies": list(STRATEGY_REGISTRY),
            "heads": ["affinity", "structure"],
        }, indent=2)

    def _build_and_train(
        self,
        name: str,
        block: str,
        block_config: dict,
        embedder: str = "input",
        embedder_config: Optional[dict] = None,
        n_blocks: int = 8,
        structure_module: str = "ipa",
        structure_module_config: Optional[dict] = None,
        strategy: str = "lora",
        strategy_config: Optional[dict] = None,
        head: str = "affinity",
        head_config: Optional[dict] = None,
        epochs: int = 10,
    ) -> str:
        from molfun.modules.builder import ModelBuilder
        from molfun.models.structure import MolfunStructureModel

        _ensure_strategy_registry()

        config = ExperimentConfig(
            name=name, embedder=embedder, embedder_config=embedder_config or {},
            block=block, block_config=block_config, n_blocks=n_blocks,
            structure_module=structure_module,
            structure_module_config=structure_module_config or {},
            strategy=strategy, strategy_config=strategy_config or {},
            head=head, head_config=head_config or {},
            epochs=epochs,
        )
        experiment = Experiment(config=config, status="running")
        t0 = time.time()

        try:
            built = ModelBuilder(
                embedder=config.embedder,
                embedder_config=config.embedder_config or None,
                block=config.block,
                block_config=config.block_config or None,
                n_blocks=config.n_blocks,
                structure_module=config.structure_module,
                structure_module_config=config.structure_module_config or None,
            ).build()

            model = MolfunStructureModel.from_custom(
                built, device=self.device,
                head=config.head, head_config=config.head_config or None,
            )

            strat_cls = STRATEGY_REGISTRY.get(config.strategy)
            if strat_cls is None:
                raise ValueError(
                    f"Unknown strategy '{config.strategy}'. "
                    f"Available: {list(STRATEGY_REGISTRY)}"
                )
            strat = strat_cls(**config.strategy_config)

            history = model.fit(
                self.train_loader, self.val_loader,
                strategy=strat, epochs=config.epochs,
            )

            experiment.history = history
            experiment.status = "completed"
            experiment.duration_s = time.time() - t0

            test_metrics = {}
            if self.test_loader is not None:
                test_metrics = self._evaluate_model(model)
            experiment.metrics = test_metrics

            self._models[experiment.id] = model

        except Exception as e:
            experiment.status = "failed"
            experiment.error = f"{type(e).__name__}: {e}"
            experiment.duration_s = time.time() - t0

        # Store in memory (the agent's memory.log_experiment handles persistence)
        self._last_experiment = experiment

        result = {
            "experiment_id": experiment.id,
            "status": experiment.status,
            "duration_s": round(experiment.duration_s, 1),
        }
        if experiment.status == "completed":
            result["best_val_loss"] = experiment.best_val_loss
            result["final_train_loss"] = experiment.final_train_loss
            result["epochs_run"] = len(experiment.history)
            if experiment.metrics:
                result["test_metrics"] = experiment.metrics
            params = built.param_summary()
            result["total_params"] = params["total"]
            result["trainable_params"] = params["trainable"]
        else:
            result["error"] = experiment.error

        return json.dumps(result, indent=2, default=str)

    def _evaluate_model(self, model) -> dict:
        """Run evaluation on test set and return metrics."""
        from molfun.helpers.training import unpack_batch, to_device

        model.adapter.eval()
        if model.head is not None:
            model.head.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_data in self.test_loader:
                batch, targets, mask = unpack_batch(batch_data)
                batch = to_device(batch, model.device)
                if targets is not None:
                    targets = targets.to(model.device)
                result = model.forward(batch, mask=mask)
                if "preds" in result and targets is not None:
                    all_preds.append(result["preds"].detach().cpu())
                    all_targets.append(targets.detach().cpu())

        if not all_preds:
            return {}

        preds = torch.cat(all_preds).squeeze()
        targets = torch.cat(all_targets).squeeze()

        metrics = {}
        metrics["test_mae"] = (preds - targets).abs().mean().item()
        metrics["test_rmse"] = ((preds - targets) ** 2).mean().sqrt().item()

        if preds.numel() > 2:
            vp = preds - preds.mean()
            vt = targets - targets.mean()
            denom = (vp.norm() * vt.norm())
            if denom > 1e-8:
                metrics["test_pearson"] = (vp * vt).sum().item() / denom.item()

        return {k: round(v, 6) for k, v in metrics.items()}

    def _get_experiment(self, experiment_id: str) -> str:
        exp = getattr(self, '_last_experiment', None)
        if exp and exp.id == experiment_id:
            return exp.to_json()
        return json.dumps({"error": f"Experiment {experiment_id} not found in tool cache"})

    def _get_journal(self) -> str:
        # This returns a placeholder; the agent loop injects actual memory
        return "Use memory context above for experiment history."

    def _save_best_model(self, path: str, experiment_id: Optional[str] = None) -> str:
        target_id = experiment_id
        if target_id and target_id in self._models:
            model = self._models[target_id]
            model.save(path)
            return json.dumps({"saved": path, "experiment_id": target_id})
        return json.dumps({"error": "Model not found. Only the most recently trained models are kept in memory."})

    def _done_handler(self, summary: str) -> str:
        self._done = True
        self._done_summary = summary
        return "Agent signaled completion."

    def get_last_experiment(self) -> Optional[Experiment]:
        """Used by the agent loop to persist experiments to memory."""
        return getattr(self, '_last_experiment', None)

    def clear_last_experiment(self) -> None:
        self._last_experiment = None
