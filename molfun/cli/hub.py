"""
CLI commands for Hugging Face Hub push/pull.
"""

from __future__ import annotations
from pathlib import Path
from typing import Annotated, Optional

import typer


def push(
    checkpoint: Annotated[Path, typer.Argument(help="Local checkpoint directory to push.")],
    repo: Annotated[str, typer.Option(help="HF Hub repo (e.g. user/model-name).")],
    token: Annotated[Optional[str], typer.Option(envvar="HF_TOKEN", help="HF API token.")] = None,
    private: Annotated[bool, typer.Option(help="Make repo private.")] = False,
    dataset_name: Annotated[Optional[str], typer.Option(help="Training dataset name for model card.")] = None,
    message: Annotated[str, typer.Option(help="Commit message.")] = "Upload Molfun model",
):
    """Push a model checkpoint to Hugging Face Hub with auto-generated model card."""
    if not checkpoint.exists():
        raise typer.BadParameter(f"Checkpoint not found: {checkpoint}")

    from molfun.tracking.hf_tracker import HuggingFaceTracker
    from molfun.tracking.model_card import generate_model_card

    typer.echo(f"Pushing {checkpoint} → {repo}...")

    tracker = HuggingFaceTracker(repo_id=repo, token=token, private=private)
    tracker.start_run(name=message)

    tracker.log_artifact(str(checkpoint), name="checkpoint")

    meta_path = checkpoint / "meta.pt"
    model_summary = {}
    if meta_path.exists():
        import torch
        meta = torch.load(meta_path, map_location="cpu", weights_only=True)
        model_summary = {"name": meta.get("name", "unknown"), "strategy": meta.get("strategy", {})}

    card = generate_model_card(
        model_summary=model_summary,
        dataset_name=dataset_name,
    )
    tracker.upload_model_card(card)
    tracker.end_run()

    typer.echo(f"Done: https://huggingface.co/{repo}")


def pull(
    repo: Annotated[str, typer.Argument(help="HF Hub repo (e.g. user/model-name).")],
    output: Annotated[Path, typer.Option("-o", help="Local output directory.")] = Path("models/"),
    token: Annotated[Optional[str], typer.Option(envvar="HF_TOKEN", help="HF API token.")] = None,
    revision: Annotated[Optional[str], typer.Option(help="Git revision (branch, tag, commit).")] = None,
):
    """Download a model from Hugging Face Hub."""
    from molfun.tracking.hf_tracker import HuggingFaceTracker

    typer.echo(f"Pulling {repo} → {output}/...")

    tracker = HuggingFaceTracker(repo_id=repo, token=token)
    local_dir = tracker.download_repo(str(output), revision=revision)

    typer.echo(f"Downloaded to {local_dir}")
    contents = list(Path(local_dir).rglob("*"))
    files = [f for f in contents if f.is_file()]
    typer.echo(f"  {len(files)} file(s):")
    for f in files[:15]:
        size_kb = f.stat().st_size / 1024
        typer.echo(f"    {f.relative_to(local_dir)}  ({size_kb:.1f} KB)")
    if len(files) > 15:
        typer.echo(f"    ... and {len(files) - 15} more")


def push_dataset(
    data_dir: Annotated[Path, typer.Argument(help="Directory with data files to upload.")],
    repo: Annotated[str, typer.Option(help="HF Hub dataset repo (e.g. user/dataset-name).")],
    token: Annotated[Optional[str], typer.Option(envvar="HF_TOKEN", help="HF API token.")] = None,
    private: Annotated[bool, typer.Option(help="Make repo private.")] = False,
    message: Annotated[str, typer.Option(help="Commit message.")] = "Upload Molfun dataset",
):
    """Push a dataset directory to Hugging Face Hub."""
    if not data_dir.exists():
        raise typer.BadParameter(f"Directory not found: {data_dir}")

    from molfun.tracking.hf_tracker import HuggingFaceTracker

    typer.echo(f"Pushing {data_dir} → {repo} (dataset)...")

    tracker = HuggingFaceTracker(
        repo_id=repo, token=token, private=private, repo_type="dataset",
    )
    tracker.start_run(name=message)
    tracker.log_artifact(str(data_dir), name="data")
    tracker.end_run()

    typer.echo(f"Done: https://huggingface.co/datasets/{repo}")
