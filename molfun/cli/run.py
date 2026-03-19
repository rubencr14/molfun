"""
CLI command for running pipeline recipes.
"""

from __future__ import annotations
from pathlib import Path
from typing import Annotated, Optional

import typer


def run(
    recipe: Annotated[Path, typer.Argument(help="Path to YAML recipe file.")],
    from_step: Annotated[Optional[str], typer.Option("--from", help="Resume from this step.")] = None,
    state_file: Annotated[Optional[Path], typer.Option("--state", help="Load state from JSON (for --from).")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print steps without executing.")] = False,
    device: Annotated[str, typer.Option(help="Override device for all steps.")] = "cpu",
    workers: Annotated[int, typer.Option("-w", help="Override download workers.")] = 4,
    verbose: Annotated[bool, typer.Option("-v", help="Verbose logging.")] = False,
):
    """Run a pipeline from a YAML recipe file."""
    import logging

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not recipe.exists():
        typer.echo(f"Recipe not found: {recipe}", err=True)
        raise typer.Exit(code=1)

    from molfun.pipelines import Pipeline

    overrides = {"device": device, "workers": workers}
    pipeline = Pipeline.from_yaml(str(recipe), **overrides)

    if dry_run:
        typer.echo(f"\nPipeline: {recipe.name}")
        typer.echo(f"Steps ({len(pipeline.steps)}):\n")
        for i, step in enumerate(pipeline.steps, 1):
            skip = " [SKIP]" if step.skip else ""
            config_preview = ""
            if step.config:
                items = [f"{k}={v}" for k, v in step.config.items() if k not in ("device", "workers")]
                if items:
                    config_preview = f"  ({', '.join(items[:5])})"
            typer.echo(f"  {i}. {step.name}{skip}{config_preview}")
        typer.echo("")
        raise typer.Exit()

    state = {}
    if state_file and state_file.exists():
        import json
        state = json.loads(state_file.read_text())
        typer.echo(f"Loaded state from {state_file} ({len(state)} keys)")

    typer.echo(f"\nRunning pipeline: {recipe.name}")
    typer.echo(f"Steps: {' → '.join(pipeline.step_names)}\n")

    try:
        if from_step:
            typer.echo(f"Resuming from step '{from_step}'...")
            result = pipeline.run_from(from_step, state)
        else:
            result = pipeline.run(state)
    except Exception as e:
        typer.echo(f"\nPipeline failed: {e}", err=True)
        raise typer.Exit(code=1)

    results = result.get("_pipeline_results", [])
    typer.echo(f"\n{'─' * 50}")
    typer.echo("  Pipeline complete")
    typer.echo(f"{'─' * 50}")
    for r in results:
        status = "SKIP" if r.skipped else ("FAIL" if r.error else "OK")
        typer.echo(f"  {r.name:20s} {status:6s} {r.elapsed_s:.1f}s")

    metrics = result.get("eval_metrics")
    if metrics:
        typer.echo(f"\n  Eval metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                typer.echo(f"    {k:20s}: {v:.6f}")
            else:
                typer.echo(f"    {k:20s}: {v}")

    typer.echo("")
