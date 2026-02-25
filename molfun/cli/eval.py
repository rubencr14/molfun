"""
CLI command for evaluating a trained model checkpoint.
"""

from __future__ import annotations
from pathlib import Path
from typing import Annotated, Optional

import typer


def eval_model(
    checkpoint: Annotated[Path, typer.Argument(help="Path to saved model checkpoint directory.")],
    pdbs: Annotated[Path, typer.Option(help="Directory with PDB/CIF test structures.")] = Path("data/structures"),
    data: Annotated[Optional[Path], typer.Option(help="CSV with pdb_id and affinity columns.")] = None,
    device: Annotated[str, typer.Option(help="Device: cuda, cpu, mps.")] = "cpu",
    max_seq_len: Annotated[int, typer.Option(help="Max sequence length.")] = 256,
    batch_size: Annotated[int, typer.Option(help="Evaluation batch size.")] = 1,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
):
    """Evaluate a trained model checkpoint on test data."""
    import json
    import torch
    from torch.utils.data import DataLoader

    if not checkpoint.exists():
        raise typer.BadParameter(f"Checkpoint not found: {checkpoint}")

    from molfun.models.structure import MolfunStructureModel

    typer.echo(f"Loading model from {checkpoint}...")
    model = MolfunStructureModel.load(str(checkpoint), device=device)

    loader = _build_eval_loader(pdbs, data, max_seq_len, batch_size)
    typer.echo(f"Evaluating on {len(loader.dataset)} samples...")

    metrics = _evaluate(model, loader)

    if json_output:
        typer.echo(json.dumps(metrics, indent=2))
    else:
        typer.echo(f"\n{'─' * 40}")
        typer.echo("  Evaluation Results")
        typer.echo(f"{'─' * 40}")
        for k, v in metrics.items():
            typer.echo(f"  {k:20s}: {v:.6f}")
        typer.echo(f"{'─' * 40}")


def _build_eval_loader(pdbs: Path, data_csv: Optional[Path], max_seq_len: int, batch_size: int):
    from torch.utils.data import DataLoader

    if data_csv and data_csv.exists():
        from molfun.data import AffinityDataset
        dataset = AffinityDataset.from_csv(str(data_csv), str(pdbs))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=AffinityDataset.collate_fn)
    elif pdbs.exists() and pdbs.is_dir():
        from molfun.data.datasets.structure import StructureDataset, collate_structure_batch
        pdb_paths = sorted(pdbs.glob("*.pdb")) + sorted(pdbs.glob("*.cif"))
        if not pdb_paths:
            raise typer.BadParameter(f"No PDB/CIF files found in {pdbs}")
        dataset = StructureDataset(pdb_paths=pdb_paths, max_seq_len=max_seq_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_structure_batch)
    else:
        raise typer.BadParameter(f"Provide --data CSV or PDB directory. Path not found: {pdbs}")


def _evaluate(model, loader) -> dict:
    import torch
    from molfun.helpers.training import unpack_batch, to_device

    model.adapter.eval()
    if model.head is not None:
        model.head.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in loader:
            batch, targets, mask = unpack_batch(batch_data)
            batch = to_device(batch, model.device)
            if targets is not None:
                targets = targets.to(model.device)
            result = model.forward(batch, mask=mask)
            if "preds" in result and targets is not None:
                all_preds.append(result["preds"].detach().cpu())
                all_targets.append(targets.detach().cpu())

    if not all_preds:
        return {"note": "No predictions produced (model may not have a prediction head)"}

    preds = torch.cat(all_preds).squeeze()
    targets = torch.cat(all_targets).squeeze()

    metrics = {
        "mae": (preds - targets).abs().mean().item(),
        "rmse": ((preds - targets) ** 2).mean().sqrt().item(),
        "n_samples": preds.numel(),
    }

    if preds.numel() > 2:
        vp = preds - preds.mean()
        vt = targets - targets.mean()
        denom = vp.norm() * vt.norm()
        if denom > 1e-8:
            metrics["pearson"] = (vp * vt).sum().item() / denom.item()

    metrics["mean_pred"] = preds.mean().item()
    metrics["std_pred"] = preds.std().item() if preds.numel() > 1 else 0.0
    metrics["mean_target"] = targets.mean().item()
    metrics["std_target"] = targets.std().item() if targets.numel() > 1 else 0.0

    return {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()}
