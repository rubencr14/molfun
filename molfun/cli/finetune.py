"""
Fine-tuning commands: structure and affinity.
"""

from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

_DEFAULT_WEIGHTS = Path.home() / ".molfun" / "weights" / "finetuning_ptm_2.pt"


class Strategy(str, Enum):
    lora      = "lora"
    partial   = "partial"
    full      = "full"
    head_only = "head_only"


def _build_strategy(
    strategy: Strategy,
    lr: float,
    rank: int,
    unfreeze: int,
    warmup: int,
    loss_fn: str = "mse",
):
    """Instantiate a FinetuneStrategy from CLI parameters."""
    common = dict(
        warmup_steps=warmup,
        scheduler="cosine",
        ema_decay=0.999,
        grad_clip=1.0,
        accumulation_steps=4,
        amp=False,
        early_stopping_patience=5,
        loss_fn=loss_fn,
    )
    if strategy == Strategy.lora:
        from molfun.training.lora import LoRAFinetune
        return LoRAFinetune(
            rank=rank,
            alpha=rank * 2,
            target_modules=["linear_q", "linear_v"],
            lr_lora=lr,
            **common,
        )
    if strategy == Strategy.partial:
        from molfun.training.partial import PartialFinetune
        return PartialFinetune(
            unfreeze_last_n=unfreeze,
            unfreeze_structure_module=True,
            lr_trunk=lr,
            **common,
        )
    if strategy == Strategy.full:
        from molfun.training.full import FullFinetune
        return FullFinetune(lr=lr, accumulation_steps=8, **{k: v for k, v in common.items() if k != "accumulation_steps"})
    if strategy == Strategy.head_only:
        from molfun.training.head_only import HeadOnlyFinetune
        return HeadOnlyFinetune(lr=lr, **common)
    raise ValueError(f"Unknown strategy: {strategy}")


# ──────────────────────────────────────────────────────────────────────
# structure
# ──────────────────────────────────────────────────────────────────────

def structure(
    pdbs:        Annotated[Path, typer.Argument(help="Directory of PDB files, or a .txt file listing their paths.")],
    strategy:    Annotated[Strategy, typer.Option(help="Fine-tuning strategy.")] = Strategy.lora,
    epochs:      Annotated[int,   typer.Option()] = 10,
    lr:          Annotated[float, typer.Option()] = 2e-4,
    rank:        Annotated[int,   typer.Option(help="LoRA rank (lora only).")] = 8,
    unfreeze:    Annotated[int,   typer.Option(help="Evoformer blocks to unfreeze (partial only).")] = 4,
    warmup:      Annotated[int,   typer.Option()] = 50,
    max_seq_len: Annotated[int,   typer.Option(help="Max residues passed to the featurizer.")] = 256,
    val_frac:    Annotated[float, typer.Option(help="Fraction of PDBs held out for validation.")] = 0.2,
    seed:        Annotated[int,   typer.Option()] = 42,
    weights:     Annotated[Path,  typer.Option(help="Pre-trained OpenFold weights.")] = _DEFAULT_WEIGHTS,
    output:      Annotated[Path,  typer.Option(help="Directory for checkpoint and history.")] = Path("runs/structure/checkpoint"),
):
    """Fine-tune OpenFold on PDB structures using FAPE loss (no affinity labels needed)."""
    import torch
    from torch.utils.data import DataLoader, Dataset, random_split
    from openfold.config import model_config
    from molfun.models.structure import MolfunStructureModel
    from molfun.backends.openfold import OpenFoldFeaturizer

    torch.manual_seed(seed)

    if not pdbs.exists():
        raise typer.BadParameter(f"Path does not exist: {pdbs}")

    if pdbs.is_dir():
        pdb_paths = sorted(pdbs.glob("*.pdb")) + sorted(pdbs.glob("*.cif"))
    elif pdbs.suffix in {".txt", ""}:
        base = pdbs.parent
        pdb_paths = [base / p.strip() for p in pdbs.read_text().splitlines() if p.strip()]
    else:
        raise typer.BadParameter(f"Expected a directory or .txt file listing paths, got: {pdbs}")

    if not pdb_paths:
        raise typer.BadParameter(f"No PDB/CIF files found in: {pdbs}")

    print(f"PDB files : {len(pdb_paths)}")

    config     = model_config("model_1_ptm")
    featurizer = OpenFoldFeaturizer(config, max_seq_len=max_seq_len)

    class _PDBDataset(Dataset):
        def __len__(self): return len(pdb_paths)
        def __getitem__(self, i): return featurizer.from_pdb(pdb_paths[i], chain_id="A")

    def _collate(batch):
        item = batch[0]
        return {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in item.items()}

    n_val   = max(1, int(len(pdb_paths) * val_frac))
    n_train = len(pdb_paths) - n_val
    train_ds, val_ds = random_split(
        _PDBDataset(), [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=_collate)
    print(f"Train     : {n_train} | Val: {n_val}")

    model = MolfunStructureModel(
        "openfold",
        config=config,
        weights=str(weights),
        device="cuda",
        head="structure",
        head_config={"loss_config": config.loss},
    )
    info = model.summary()["adapter"]
    print(f"Params    : {info['total']:,} total | {info['trainable']:,} trainable")

    strat = _build_strategy(strategy, lr, rank, unfreeze, warmup, loss_fn="openfold")
    strat.fit(model, train_loader, val_loader, epochs=epochs, verbose=True)

    output.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    torch.save(strat, output / "history.pt")
    print(f"Checkpoint: {output}")


# ──────────────────────────────────────────────────────────────────────
# affinity
# ──────────────────────────────────────────────────────────────────────

def affinity(
    pdbs:     Annotated[Path,          typer.Argument(help="Directory with PDB/mmCIF structure files.")],
    data:     Annotated[Optional[Path], typer.Option(help="CSV with pdb_id and affinity columns.")] = None,
    demo:     Annotated[bool,           typer.Option(help="Run with synthetic data (no CSV needed).")] = False,
    strategy: Annotated[Strategy,       typer.Option()] = Strategy.lora,
    epochs:   Annotated[int,   typer.Option()] = 20,
    lr:       Annotated[float, typer.Option()] = 1e-4,
    rank:     Annotated[int,   typer.Option(help="LoRA rank (lora only).")] = 8,
    unfreeze: Annotated[int,   typer.Option(help="Evoformer blocks to unfreeze (partial only).")] = 4,
    warmup:   Annotated[int,   typer.Option()] = 100,
    loss:     Annotated[str,   typer.Option(help="Loss function: mse | mae | huber | pearson.")] = "mse",
    val_frac: Annotated[float, typer.Option()] = 0.1,
    seed:     Annotated[int,   typer.Option()] = 42,
    weights:  Annotated[Path,  typer.Option(help="Pre-trained OpenFold weights.")] = _DEFAULT_WEIGHTS,
    output:   Annotated[Path,  typer.Option(help="Directory for checkpoint and history.")] = Path("runs/affinity/checkpoint"),
):
    """Fine-tune OpenFold on binding affinity data (ΔG / pKd regression)."""
    if not demo and data is None:
        raise typer.BadParameter("--data is required unless --demo is set.")

    import torch
    from torch.utils.data import DataLoader
    from openfold.config import model_config
    from molfun.models.structure import MolfunStructureModel

    torch.manual_seed(seed)
    config = model_config("model_1_ptm")

    if demo:
        from tests.models.test_openfold_real import DummyOpenFoldDataset, _collate
        n = 8
        train_loader = DataLoader(DummyOpenFoldDataset(n - 2, seq_len=32, n_msa=4), batch_size=1, collate_fn=_collate)
        val_loader   = DataLoader(DummyOpenFoldDataset(2,     seq_len=32, n_msa=4), batch_size=1, collate_fn=_collate)
        print("Demo mode : 6 train / 2 val (synthetic)")
    else:
        from molfun.data import AffinityDataset, DataSplitter
        dataset = AffinityDataset.from_csv(str(data), str(pdbs))
        train_ds, val_ds, _ = DataSplitter.random(dataset, val_frac=val_frac, test_frac=0.1, seed=seed)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  collate_fn=AffinityDataset.collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=AffinityDataset.collate_fn)
        print(f"Dataset   : {len(train_ds)} train | {len(val_ds)} val")

    model = MolfunStructureModel(
        "openfold",
        config=config,
        weights=str(weights),
        device="cuda",
        head="affinity",
        head_config={"single_dim": 384, "hidden_dim": 128},
    )
    info = model.summary()["adapter"]
    print(f"Params    : {info['total']:,} total | {info['trainable']:,} trainable")

    strat = _build_strategy(strategy, lr, rank, unfreeze, warmup, loss_fn=loss)
    history = strat.fit(model, train_loader, val_loader, epochs=epochs, verbose=True)

    output.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    torch.save(history, output / "history.pt")
    print(f"Checkpoint: {output}")
