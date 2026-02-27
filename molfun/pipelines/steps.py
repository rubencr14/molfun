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


# ==================================================================
# Dataset loading (shared between head training and classical ML)
# ==================================================================

def load_dataset_step(state: StateDict) -> StateDict:
    """
    Load a PropertyDataset into the pipeline state.

    This step is **optional** for structure fine-tuning (targets = 3D coords
    inside the PDB), but **required** for property heads and classical ML.

    Uses ``PropertyDataset`` as the unified data wrapper, supporting
    CSV files, inline data, and matching fetched PDB paths with labels.

    Config keys:
        source (str): One of "csv", "inline", "fetch_csv". Default "csv".

        --- CSV source ---
        targets_file (str):  Path to CSV/TSV file.
        sequence_col (str):  Column with sequences (default "sequence").
        target_col (str):    Column with target values (default "target").
        pdb_id_col (str):    Column with PDB IDs (default "pdb_id").
        sep (str):           Separator (auto-detected if omitted).
        pdb_dir (str):       If given, resolves PDB paths from pdb_id column.
        pdb_fmt (str):       File extension for PDBs (default "cif").

        --- Inline source ---
        sequences (list[str]):  Protein sequences.
        targets (list[float]):  Target values.

        --- fetch_csv source ---
        targets_file (str):  CSV with labels.
        target_col (str):    Column with target values.
        pdb_id_col (str):    Column with PDB IDs.
        (Uses pdb_paths from a previous fetch step in the state.)

    Produces:
        state["dataset"]:    PropertyDataset instance.
        state["sequences"]:  list of sequences.
        state["y"]:          np.ndarray of targets.
        state["pdb_ids"]:    list of PDB IDs (if available).
        state["pdb_paths"]:  list of paths (if available).
    """
    from molfun.data.datasets.property import PropertyDataset

    source = state.get("source", "csv")

    if source == "csv":
        ds = PropertyDataset.from_csv(
            csv_path=state["targets_file"],
            sequence_col=state.get("sequence_col", "sequence"),
            target_col=state.get("target_col", "target"),
            pdb_id_col=state.get("pdb_id_col", "pdb_id"),
            sep=state.get("sep"),
            pdb_dir=state.get("pdb_dir"),
            pdb_fmt=state.get("pdb_fmt", "cif"),
        )
    elif source == "inline":
        ds = PropertyDataset.from_inline(
            sequences=state.get("sequences"),
            targets=state.get("targets"),
            pdb_paths=state.get("pdb_paths"),
            target_name=state.get("target_col", "target"),
        )
    elif source == "fetch_csv":
        pdb_paths = state.get("pdb_paths", [])
        if not pdb_paths:
            raise ValueError("source='fetch_csv' requires pdb_paths from a fetch step")
        ds = PropertyDataset.from_fetch_and_csv(
            pdb_paths=pdb_paths,
            csv_path=state["targets_file"],
            target_col=state.get("target_col", "target"),
            pdb_id_col=state.get("pdb_id_col", "pdb_id"),
            sep=state.get("sep"),
        )
    else:
        raise ValueError(
            f"Unknown source '{source}'. Available: 'csv', 'inline', 'fetch_csv'"
        )

    out = {**state, "dataset": ds}
    out.update(ds.to_dict())
    return out


# Backward-compatible alias
load_targets_step = load_dataset_step


# ==================================================================
# Sklearn / classical ML steps
# ==================================================================

# ------------------------------------------------------------------
# Featurize
# ------------------------------------------------------------------

def featurize_step(state: StateDict) -> StateDict:
    """
    Extract numerical features from protein sequences.

    Config keys:
        features (list[str]): Feature names (default: DEFAULT_FEATURES).
            See ``molfun.ml.features.AVAILABLE_FEATURES``.

    Requires:
        state["sequences"]: list of protein sequences (str).
        OR state["pdb_paths"]: PDB/CIF paths (sequences extracted automatically).

    Produces:
        state["X"]: np.ndarray feature matrix [N, D].
        state["feature_names"]: list of feature column names.
        state["sequences"]: list of sequences (if extracted from PDBs).
    """
    from molfun.ml.features import ProteinFeaturizer

    features = state.get("features")
    featurizer = ProteinFeaturizer(features=features)

    sequences = state.get("sequences")
    if sequences is None:
        pdb_paths = state.get("pdb_paths")
        if pdb_paths is None:
            raise ValueError("featurize_step requires 'sequences' or 'pdb_paths'")
        sequences = _extract_sequences(pdb_paths)

    X = featurizer.fit_transform(sequences)

    return {
        **state,
        "X": X,
        "sequences": sequences,
        "feature_names": featurizer.feature_names,
        "n_features": featurizer.n_features,
        "_featurizer": featurizer,
    }


def _extract_sequences(pdb_paths: list) -> list[str]:
    """Extract sequences from PDB/CIF files using molfun parsers."""
    from molfun.data.parsers import auto_parser

    sequences = []
    for p in pdb_paths:
        parser = auto_parser(str(p))
        parsed = parser.parse_file(str(p))
        sequences.append(parsed.sequence)
    return sequences


# ------------------------------------------------------------------
# Split for sklearn
# ------------------------------------------------------------------

def split_sklearn_step(state: StateDict) -> StateDict:
    """
    Split features + targets into train/test for sklearn.

    Config keys:
        val_frac (float):   Validation fraction (default 0.2).
        seed (int):         Random seed (default 42).

    Requires:
        state["X"]: feature matrix.
        state["y"]: target array.

    Produces:
        state["X_train"], state["X_test"], state["y_train"], state["y_test"]
        state["sequences_train"], state["sequences_test"] (if sequences present)
    """
    from sklearn.model_selection import train_test_split
    import numpy as np

    X = state["X"]
    y = np.asarray(state["y"])
    val_frac = state.get("val_frac", 0.2)
    seed = state.get("seed", 42)

    sequences = state.get("sequences")

    if sequences is not None:
        X_tr, X_te, y_tr, y_te, seq_tr, seq_te = train_test_split(
            X, y, sequences, test_size=val_frac, random_state=seed,
        )
        return {
            **state,
            "X_train": X_tr, "X_test": X_te,
            "y_train": y_tr, "y_test": y_te,
            "sequences_train": seq_tr, "sequences_test": seq_te,
        }
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=val_frac, random_state=seed,
        )
        return {
            **state,
            "X_train": X_tr, "X_test": X_te,
            "y_train": y_tr, "y_test": y_te,
        }


# ------------------------------------------------------------------
# Train sklearn
# ------------------------------------------------------------------

def train_sklearn_step(state: StateDict) -> StateDict:
    """
    Train a classical ML model (sklearn-compatible).

    Config keys:
        estimator (str):     "random_forest", "svm", "linear", etc.
        task (str):          "regression" or "classification" (default "regression").
        scale (bool):        Apply StandardScaler (default True).
        features (list[str]): Feature names (only if no featurizer in state).
        **:                  Extra params passed to the sklearn estimator.

    Requires:
        state["X_train"], state["y_train"] (pre-featurized)
        OR state["sequences_train"], state["y_train"] (featurized on-the-fly)

    Produces:
        state["model"]: fitted ProteinRegressor or ProteinClassifier.
    """
    estimator = state.get("estimator", "random_forest")
    task = state.get("task", "regression")
    scale = state.get("scale", True)
    features = state.get("features")

    estimator_params = {}
    for k in ("n_estimators", "max_depth", "kernel", "C", "gamma",
              "alpha", "max_iter", "n_neighbors", "learning_rate"):
        if k in state:
            estimator_params[k] = state[k]

    if task == "classification":
        from molfun.ml.estimators import ProteinClassifier
        model = ProteinClassifier(
            estimator=estimator, features=features,
            scale=scale, **estimator_params,
        )
    else:
        from molfun.ml.estimators import ProteinRegressor
        model = ProteinRegressor(
            estimator=estimator, features=features,
            scale=scale, **estimator_params,
        )

    sequences_train = state.get("sequences_train")
    X_train = state.get("X_train")
    y_train = state["y_train"]

    if sequences_train is not None:
        model.fit(sequences_train, y_train)
    elif X_train is not None:
        model._fit_precomputed(X_train, y_train)
    else:
        raise ValueError("train_sklearn_step requires 'X_train' or 'sequences_train'")

    return {**state, "model": model}


# ------------------------------------------------------------------
# Eval sklearn
# ------------------------------------------------------------------

def eval_sklearn_step(state: StateDict) -> StateDict:
    """
    Evaluate a trained sklearn model.

    Requires:
        state["model"]: fitted ProteinRegressor or ProteinClassifier.
        state["X_test"] + state["y_test"]
        OR state["sequences_test"] + state["y_test"]

    Produces:
        state["eval_metrics"]: dict of metrics.
    """
    import numpy as np

    model = state["model"]
    sequences_test = state.get("sequences_test")
    X_test_features = state.get("X_test")
    y_test = np.asarray(state["y_test"])

    if sequences_test is not None:
        X_test = sequences_test
    elif X_test_features is not None:
        X_test = X_test_features
    else:
        raise ValueError("eval_sklearn_step requires 'X_test' or 'sequences_test'")

    if hasattr(model, "evaluate"):
        metrics = model.evaluate(X_test, y_test)
    else:
        preds = model.predict(X_test)
        metrics = {
            "score": float(model.score(X_test, y_test)),
            "n_samples": len(y_test),
        }

    top = []
    if hasattr(model, "top_features"):
        top = model.top_features(10)
    if top:
        metrics["top_features"] = {name: round(imp, 4) for name, imp in top}

    return {**state, "eval_metrics": metrics}


# ------------------------------------------------------------------
# Save sklearn
# ------------------------------------------------------------------

def save_sklearn_step(state: StateDict) -> StateDict:
    """
    Save a trained sklearn model to disk.

    Config keys:
        model_path (str): Output path (default "runs/model.joblib").

    Requires:
        state["model"]

    Produces:
        state["model_path"]
    """
    model = state["model"]
    model_path = state.get("model_path", "runs/model.joblib")

    from molfun.ml.io import save_model
    save_model(model, model_path)

    return {**state, "model_path": model_path}


# ==================================================================
# Property head steps (backbone embeddings + head)
# ==================================================================

def extract_embeddings_step(state: StateDict) -> StateDict:
    """
    Extract embeddings from a structure model backbone.

    Config keys:
        backbone (str):   Model name (default "openfold").
        layer (str):      Which layer ("last", "middle", etc.).
        pooling (str):    "mean", "max", or "cls" (default "mean").
        device (str):     "cpu" or "cuda" (default "cpu").
        batch_size (int): Batch size for inference (default 4).

    Requires:
        state["pdb_paths"]: list of PDB/CIF file paths.

    Produces:
        state["embeddings"]: np.ndarray [N, embedding_dim].
    """
    from molfun.ml.heads import extract_embeddings

    pdb_paths = state.get("pdb_paths")
    if pdb_paths is None:
        raise ValueError("extract_embeddings_step requires 'pdb_paths'")

    embeddings = extract_embeddings(
        pdb_paths,
        backbone=state.get("backbone", state.get("model_name", "openfold")),
        layer=state.get("layer", "last"),
        pooling=state.get("pooling", "mean"),
        device=state.get("device", "cpu"),
        batch_size=state.get("batch_size", 4),
    )

    return {**state, "embeddings": embeddings}


def train_head_step(state: StateDict) -> StateDict:
    """
    Train a property prediction head on backbone embeddings.

    Config keys:
        head_type (str):      "mlp", "linear", "rf", "svm" (default "mlp").
        task (str):           "regression" or "classification".
        hidden_dims (list):   MLP hidden layers (default [256, 128]).
        epochs (int):         MLP training epochs (default 100).
        lr (float):           MLP learning rate (default 1e-3).
        device (str):         "cpu" or "cuda".
        **:                   Extra params for the head constructor.

    Requires:
        state["embeddings"] or state["pdb_paths"]
        state["y_train"] (targets for training split)
        OR state["y"] (if no split done yet)

    Produces:
        state["model"]: fitted PropertyHead.
    """
    from molfun.ml.heads import PropertyHead

    head_type = state.get("head_type", "mlp")
    task = state.get("task", "regression")
    backbone = state.get("backbone", state.get("model_name", "openfold"))
    device = state.get("device", "cpu")

    head_params = {}
    for k in ("hidden_dims", "epochs", "lr", "batch_size",
              "n_estimators", "n_classes", "kernel", "C"):
        if k in state:
            head_params[k] = state[k]

    head = PropertyHead(
        backbone=backbone,
        head_type=head_type,
        task=task,
        hidden_dims=head_params.pop("hidden_dims", [256, 128]),
        device=device,
        **head_params,
    )

    embeddings = state.get("embeddings_train", state.get("embeddings"))
    y = state.get("y_train", state.get("y"))
    pdb_paths = state.get("pdb_paths_train", state.get("pdb_paths"))

    if y is None:
        raise ValueError("train_head_step requires 'y_train' or 'y'")

    head.fit(pdb_paths=pdb_paths or [], y=y, embeddings=embeddings)
    return {**state, "model": head}


def eval_head_step(state: StateDict) -> StateDict:
    """
    Evaluate a trained property head.

    Requires:
        state["model"]: fitted PropertyHead.
        state["embeddings_test"] or state["pdb_paths_test"]
        state["y_test"]

    Produces:
        state["eval_metrics"]: dict of metrics.
    """
    import numpy as np

    model = state["model"]
    y_test = np.asarray(state["y_test"])

    embeddings = state.get("embeddings_test")
    pdb_paths = state.get("pdb_paths_test")

    preds = model.predict(pdb_paths=pdb_paths, embeddings=embeddings)
    preds = np.asarray(preds, dtype=np.float64)

    if model.task == "regression":
        mae = float(np.abs(preds - y_test).mean())
        rmse = float(np.sqrt(((preds - y_test) ** 2).mean()))
        ss_res = np.sum((y_test - preds) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = float(1 - ss_res / max(ss_tot, 1e-8))
        metrics = {"mae": mae, "rmse": rmse, "r2": r2, "n_samples": len(y_test)}
    else:
        accuracy = float((preds == y_test).mean())
        metrics = {"accuracy": accuracy, "n_samples": len(y_test)}

    return {**state, "eval_metrics": metrics}
