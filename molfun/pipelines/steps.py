"""
Built-in pipeline step functions.

Each step is a plain function ``dict -> dict``.  It receives the merged
pipeline state + its own config, does work, and returns the updated state.

Users can write their own steps following the same contract.

Available steps:
    fetch_step      Download structures from RCSB
    split_step      Train/val/test split
    build_dataset_step  Create StructureDataset + DataLoaders
    train_step      Fine-tune with a chosen strategy
    eval_step       Evaluate on test set
    save_step       Save model checkpoint
    push_step       Push to Hugging Face Hub
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

StateDict = dict[str, Any]


# ------------------------------------------------------------------
# Fetch
# ------------------------------------------------------------------

def fetch_step(state: StateDict) -> StateDict:
    """
    Download PDB structures from RCSB.

    Config keys:
        collection (str):    Named collection (e.g. "kinases_human").
        pfam_id (str):       Pfam accession.
        ec_number (str):     EC number.
        go_id (str):         GO term.
        taxonomy_id (int):   NCBI taxonomy ID.
        keyword (str):       Free-text search.
        max_structures (int):  Max structures (default 500).
        resolution_max (float): Max resolution (default 3.0).
        deduplicate (bool):  Deduplicate by sequence (default False).
        identity (float):    Sequence identity threshold (default 0.3).
        fmt (str):           "cif" or "pdb" (default "cif").
        output_dir (str):    Cache/output directory (default "data/structures").
        workers (int):       Parallel download threads (default 4).
        progress (bool):     Show progress bar (default True).

    Produces:
        state["pdb_paths"]: list of downloaded file paths.
        state["n_structures"]: number of structures.
    """
    from molfun.data.sources.pdb import PDBFetcher

    output_dir = state.get("output_dir", "data/structures")
    fmt = state.get("fmt", "cif")
    workers = state.get("workers", 4)
    progress = state.get("progress", True)
    max_s = state.get("max_structures", 500)
    res = state.get("resolution_max", 3.0)
    dedup = state.get("deduplicate", False)
    identity = state.get("identity", 0.3)

    fetcher = PDBFetcher(
        cache_dir=output_dir, fmt=fmt,
        workers=workers, progress=progress,
    )

    collection = state.get("collection")
    if collection:
        from molfun.data.collections import fetch_collection
        paths = fetch_collection(
            collection,
            cache_dir=output_dir, fmt=fmt,
            max_structures=max_s, resolution_max=res,
            deduplicate=dedup, identity=identity,
            workers=workers, progress=progress,
        )
    else:
        pdb_ids = fetcher.search_ids(
            pfam_id=state.get("pfam_id"),
            ec_number=state.get("ec_number"),
            go_id=state.get("go_id"),
            taxonomy_id=state.get("taxonomy_id"),
            keyword=state.get("keyword"),
            max_results=max_s,
            resolution_max=res,
        )
        if dedup and pdb_ids:
            from molfun.data.sources.pdb import deduplicate_by_sequence
            pdb_ids = deduplicate_by_sequence(pdb_ids, identity=identity)
        paths = fetcher.fetch(pdb_ids)

    return {**state, "pdb_paths": paths, "n_structures": len(paths)}


# ------------------------------------------------------------------
# Split
# ------------------------------------------------------------------

def split_step(state: StateDict) -> StateDict:
    """
    Split structures into train/val/test sets.

    Config keys:
        val_frac (float):   Validation fraction (default 0.1).
        test_frac (float):  Test fraction (default 0.1).
        split_method (str): "random" (default), "by_family", "temporal".
        seed (int):         Random seed (default 42).

    Requires:
        state["pdb_paths"]: list of paths.

    Produces:
        state["train_paths"], state["val_paths"], state["test_paths"]
    """
    pdb_paths = state["pdb_paths"]
    val_frac = state.get("val_frac", 0.1)
    test_frac = state.get("test_frac", 0.1)
    seed = state.get("seed", 42)

    import random
    rng = random.Random(seed)
    paths = list(pdb_paths)
    rng.shuffle(paths)

    n = len(paths)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    train_paths = paths[:n_train]
    val_paths = paths[n_train : n_train + n_val]
    test_paths = paths[n_train + n_val :]

    return {
        **state,
        "train_paths": train_paths,
        "val_paths": val_paths,
        "test_paths": test_paths,
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "n_test": len(test_paths),
    }


# ------------------------------------------------------------------
# Build Dataset
# ------------------------------------------------------------------

def build_dataset_step(state: StateDict) -> StateDict:
    """
    Build StructureDataset and DataLoaders from split paths.

    Config keys:
        max_seq_len (int):  Max sequence length (default 256).
        batch_size (int):   Batch size (default 1).

    Requires:
        state["train_paths"], state["val_paths"]

    Produces:
        state["train_loader"], state["val_loader"]
        Optionally state["test_loader"] if test_paths present.
    """
    import torch
    from torch.utils.data import DataLoader
    from molfun.data.datasets.structure import StructureDataset, collate_structure_batch

    max_seq_len = state.get("max_seq_len", 256)
    batch_size = state.get("batch_size", 1)

    train_paths = [Path(p) for p in state["train_paths"]]
    val_paths = [Path(p) for p in state["val_paths"]]

    train_ds = StructureDataset(pdb_paths=train_paths, max_seq_len=max_seq_len)
    val_ds = StructureDataset(pdb_paths=val_paths, max_seq_len=max_seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_structure_batch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_structure_batch,
    )

    result = {
        **state,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }

    test_paths = state.get("test_paths")
    if test_paths:
        test_ds = StructureDataset(
            pdb_paths=[Path(p) for p in test_paths],
            max_seq_len=max_seq_len,
        )
        result["test_loader"] = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_structure_batch,
        )

    return result


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------

def train_step(state: StateDict) -> StateDict:
    """
    Fine-tune a model using a configured strategy.

    Config keys:
        strategy (str):     "lora", "partial", "full", "head_only" (default "lora").
        epochs (int):       Number of epochs (default 10).
        lr (float):         Learning rate (default 2e-4).
        rank (int):         LoRA rank (default 8).
        unfreeze (int):     Blocks to unfreeze for partial (default 4).
        warmup (int):       Warmup steps (default 50).
        loss_fn (str):      Loss function name (default "openfold").
        device (str):       "cuda", "cpu", "mps" (default "cpu").
        weights (str):      Path to pre-trained weights.
        model_name (str):   Backend name (default "openfold").
        head (str):         Prediction head (default "structure").

    Requires:
        state["train_loader"], state["val_loader"]

    Produces:
        state["model"], state["training_history"]
    """
    strategy_name = state.get("strategy", "lora")
    epochs = state.get("epochs", 10)
    lr = state.get("lr", 2e-4)
    rank = state.get("rank", 8)
    unfreeze = state.get("unfreeze", 4)
    warmup = state.get("warmup", 50)
    loss_fn = state.get("loss_fn", "openfold")
    device = state.get("device", "cpu")

    model = state.get("model")
    if model is None:
        from molfun.models.structure import MolfunStructureModel
        weights = state.get("weights", str(Path.home() / ".molfun" / "weights" / "finetuning_ptm_2.pt"))
        model_name = state.get("model_name", "openfold")
        head = state.get("head", "structure")

        model = MolfunStructureModel(
            model_name,
            weights=weights,
            device=device,
            head=head,
        )

    strategy = _build_strategy(strategy_name, lr, rank, unfreeze, warmup, loss_fn)
    history = strategy.fit(
        model,
        state["train_loader"],
        state.get("val_loader"),
        epochs=epochs,
        verbose=True,
    )

    return {**state, "model": model, "training_history": history}


def _build_strategy(name: str, lr: float, rank: int, unfreeze: int, warmup: int, loss_fn: str):
    """Instantiate a FinetuneStrategy by name."""
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
    if name == "lora":
        from molfun.training.lora import LoRAFinetune
        return LoRAFinetune(
            rank=rank, alpha=rank * 2,
            target_modules=["linear_q", "linear_v"],
            lr_lora=lr, **common,
        )
    if name == "partial":
        from molfun.training.partial import PartialFinetune
        return PartialFinetune(
            unfreeze_last_n=unfreeze,
            unfreeze_structure_module=True,
            lr_trunk=lr, **common,
        )
    if name == "full":
        from molfun.training.full import FullFinetune
        return FullFinetune(
            lr=lr, accumulation_steps=8,
            **{k: v for k, v in common.items() if k != "accumulation_steps"},
        )
    if name == "head_only":
        from molfun.training.head_only import HeadOnlyFinetune
        return HeadOnlyFinetune(lr=lr, **common)
    raise ValueError(f"Unknown strategy: {name}")


# ------------------------------------------------------------------
# Eval
# ------------------------------------------------------------------

def eval_step(state: StateDict) -> StateDict:
    """
    Evaluate the trained model on test data.

    Config keys:
        device (str): Device (default "cpu").

    Requires:
        state["model"]
        state["test_loader"] or state["val_loader"]

    Produces:
        state["eval_metrics"]
    """
    import torch
    from molfun.helpers.training import unpack_batch, to_device

    model = state["model"]
    loader = state.get("test_loader", state.get("val_loader"))
    if loader is None:
        raise ValueError("eval_step requires 'test_loader' or 'val_loader' in state")

    model.adapter.eval()
    if model.head is not None:
        model.head.eval()

    all_preds, all_targets = [], []

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
        return {**state, "eval_metrics": {"note": "No predictions (no prediction head)"}}

    preds = torch.cat(all_preds).squeeze()
    targets = torch.cat(all_targets).squeeze()

    metrics: dict[str, Any] = {
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

    return {**state, "eval_metrics": metrics}


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

def save_step(state: StateDict) -> StateDict:
    """
    Save model checkpoint to disk.

    Config keys:
        checkpoint_dir (str): Output directory (default "runs/checkpoint").

    Requires:
        state["model"]

    Produces:
        state["checkpoint_path"]
    """
    model = state["model"]
    ckpt_dir = state.get("checkpoint_dir", "runs/checkpoint")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    model.save(ckpt_dir)

    history = state.get("training_history")
    if history:
        import torch
        torch.save(history, Path(ckpt_dir) / "history.pt")

    return {**state, "checkpoint_path": ckpt_dir}


# ------------------------------------------------------------------
# Push to Hub
# ------------------------------------------------------------------

def push_step(state: StateDict) -> StateDict:
    """
    Push model to Hugging Face Hub.

    Config keys:
        repo (str):      HF repo ID (e.g. "user/model-name"). Required.
        token (str):     HF API token (or set HF_TOKEN env var).
        private (bool):  Make repo private (default False).
        message (str):   Commit message.

    Requires:
        state["checkpoint_path"] or state["model"]
    """
    from molfun.tracking.hf_tracker import HuggingFaceTracker

    repo = state.get("repo")
    if not repo:
        raise ValueError("push_step requires 'repo' in config (e.g. 'user/model-name')")

    ckpt = state.get("checkpoint_path")
    if not ckpt:
        raise ValueError("push_step requires 'checkpoint_path' — run save_step first")

    token = state.get("token")
    private = state.get("private", False)
    message = state.get("message", "Upload Molfun pipeline model")

    tracker = HuggingFaceTracker(repo_id=repo, token=token, private=private)
    tracker.start_run(name=message)
    tracker.push_checkpoint(str(ckpt), commit_message=message)

    return {**state, "hub_repo": repo}
