"""
Lightweight composable pipeline for end-to-end protein ML workflows.

A pipeline is an ordered list of steps where each step is a plain
function ``dict -> dict``.  State flows through steps and is
optionally checkpointed between them.

Usage::

    from molfun.pipelines import Pipeline, PipelineStep

    pipeline = Pipeline([
        PipelineStep("fetch",  fetch_step,  config={"ec_number": "2.7.11"}),
        PipelineStep("split",  split_step,  config={"val_frac": 0.1}),
        PipelineStep("train",  train_step,  config={"epochs": 20}),
        PipelineStep("eval",   eval_step),
    ])

    result = pipeline.run()
    result = pipeline.run_from("train", saved_state)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
import json
import logging
import time

logger = logging.getLogger(__name__)

StateDict = dict[str, Any]
StepFn = Callable[[StateDict], StateDict]


@dataclass
class PipelineStep:
    """A single pipeline step: a callable + static config."""
    name: str
    fn: StepFn
    config: dict[str, Any] = field(default_factory=dict)
    skip: bool = False


@dataclass
class StepResult:
    """Outcome of a single step execution."""
    name: str
    elapsed_s: float
    skipped: bool = False
    error: Optional[str] = None


class Pipeline:
    """
    Composable, checkpointed pipeline.

    Args:
        steps: Ordered list of ``PipelineStep`` instances.
        checkpoint_dir: If set, state is saved to JSON after each step.
        hooks: Optional dict with ``on_step_start(name, state)`` and/or
               ``on_step_end(name, state, result)`` callbacks.
    """

    def __init__(
        self,
        steps: list[PipelineStep],
        checkpoint_dir: Optional[str] = None,
        hooks: Optional[dict[str, Callable]] = None,
    ):
        self.steps = steps
        self.checkpoint_dir = checkpoint_dir
        self.hooks = hooks or {}

        names = [s.name for s in steps]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate step names: {names}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, state: Optional[StateDict] = None) -> StateDict:
        """Execute all steps sequentially, passing state through each."""
        return self._run_from(0, state or {})

    def run_from(self, step_name: str, state: StateDict) -> StateDict:
        """Resume execution from a named step (inclusive)."""
        idx = self._step_index(step_name)
        return self._run_from(idx, state)

    def dry_run(self, state: Optional[StateDict] = None) -> list[str]:
        """Return the names of steps that would execute (not skipped)."""
        return [s.name for s in self.steps if not s.skip]

    @property
    def step_names(self) -> list[str]:
        return [s.name for s in self.steps]

    def describe(self) -> list[dict]:
        """Return a serializable description of the pipeline."""
        return [
            {
                "name": s.name,
                "fn": f"{s.fn.__module__}.{s.fn.__qualname__}",
                "config": s.config,
                "skip": s.skip,
            }
            for s in self.steps
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_from(self, start_idx: int, state: StateDict) -> StateDict:
        results: list[StepResult] = []

        for step in self.steps[start_idx:]:
            if step.skip:
                logger.info("[pipeline] Skipping '%s'", step.name)
                results.append(StepResult(name=step.name, elapsed_s=0, skipped=True))
                continue

            merged = {**state, **step.config}
            self._call_hook("on_step_start", step.name, merged)
            logger.info("[pipeline] Running '%s'...", step.name)

            t0 = time.time()
            try:
                state = step.fn(merged)
            except Exception as exc:
                elapsed = time.time() - t0
                res = StepResult(name=step.name, elapsed_s=elapsed, error=str(exc))
                results.append(res)
                self._call_hook("on_step_end", step.name, state, res)
                logger.error("[pipeline] Step '%s' failed after %.1fs: %s", step.name, elapsed, exc)
                state["_pipeline_error"] = {"step": step.name, "error": str(exc)}
                state["_pipeline_results"] = results
                raise
            elapsed = time.time() - t0

            res = StepResult(name=step.name, elapsed_s=elapsed)
            results.append(res)
            self._call_hook("on_step_end", step.name, state, res)
            logger.info("[pipeline] '%s' done (%.1fs)", step.name, elapsed)

            if self.checkpoint_dir:
                self._save_checkpoint(step.name, state)

        state["_pipeline_results"] = results
        return state

    def _step_index(self, name: str) -> int:
        for i, s in enumerate(self.steps):
            if s.name == name:
                return i
        available = ", ".join(s.name for s in self.steps)
        raise ValueError(f"Step '{name}' not found. Available: {available}")

    def _save_checkpoint(self, step_name: str, state: StateDict) -> None:
        ckpt_dir = Path(self.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        serializable = {}
        for k, v in state.items():
            if k.startswith("_"):
                continue
            try:
                json.dumps(v)
                serializable[k] = v
            except (TypeError, ValueError):
                serializable[k] = f"<{type(v).__name__}>"

        path = ckpt_dir / f"state_after_{step_name}.json"
        path.write_text(json.dumps(serializable, indent=2, default=str))
        logger.debug("[pipeline] Checkpoint saved: %s", path)

    def _call_hook(self, hook_name: str, *args) -> None:
        fn = self.hooks.get(hook_name)
        if fn is not None:
            try:
                fn(*args)
            except Exception as exc:
                logger.warning("[pipeline] Hook '%s' failed: %s", hook_name, exc)

    # ------------------------------------------------------------------
    # YAML recipe loading
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> Pipeline:
        """
        Load a pipeline from a YAML recipe file.

        Recipe format::

            checkpoint_dir: runs/my_experiment  # optional

            steps:
              - name: fetch
                fn: molfun.pipelines.steps.fetch_step
                config:
                  ec_number: "2.7.11"
                  max_structures: 200

              - name: train
                fn: molfun.pipelines.steps.train_step
                config:
                  strategy: lora
                  epochs: 20

        Args:
            path: Path to YAML file.
            **overrides: Key-value pairs merged into every step's config
                         (useful for CLI overrides like ``device=cuda``).
        """
        import yaml

        raw = Path(path).read_text()
        recipe = yaml.safe_load(raw)

        steps = []
        for entry in recipe.get("steps", []):
            fn = _import_callable(entry["fn"])
            config = entry.get("config", {})
            if overrides:
                config = {**config, **overrides}
            steps.append(PipelineStep(
                name=entry["name"],
                fn=fn,
                config=config,
                skip=entry.get("skip", False),
            ))

        return cls(
            steps=steps,
            checkpoint_dir=recipe.get("checkpoint_dir"),
        )


def _import_callable(dotted_path: str) -> Callable:
    """Import a function from a dotted module path like 'molfun.pipelines.steps.fetch_step'."""
    module_path, _, func_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid callable path: {dotted_path}")
    import importlib
    module = importlib.import_module(module_path)
    fn = getattr(module, func_name, None)
    if fn is None:
        raise ImportError(f"Cannot find '{func_name}' in '{module_path}'")
    if not callable(fn):
        raise TypeError(f"'{dotted_path}' is not callable")
    return fn
