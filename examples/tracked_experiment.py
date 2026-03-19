#!/usr/bin/env python
"""
Run a tracked experiment with multiple logging backends.

Molfun's CompositeTracker lets you log to several services simultaneously:
  - Console (always — see what's happening)
  - Weights & Biases (team dashboards, sweeps)
  - Comet ML (comparisons, panels)
  - MLflow (on-prem, reproducibility)
  - Langfuse (LLM agent tracing, if using agents)
  - Hugging Face Hub (push model + metrics as artifacts)

You only configure once; every tracker receives the same events.

Requires:
  pip install molfun[wandb,comet,mlflow,hub]  # pick the ones you use
"""

import os
from pathlib import Path

import torch

from molfun.tracking import (
    ConsoleTracker,
    CompositeTracker,
)

# ── 1. Configure trackers ────────────────────────────────────────────

trackers = [ConsoleTracker(verbose=True)]

# W&B — set WANDB_API_KEY or run `wandb login`
if os.getenv("WANDB_API_KEY"):
    from molfun.tracking import WandbTracker
    trackers.append(WandbTracker(
        project="molfun-kinase",
        name="lora-r8-pairformer",
        tags=["kinase", "lora", "pairformer"],
    ))
    print("[+] Weights & Biases enabled")

# Comet — set COMET_API_KEY
if os.getenv("COMET_API_KEY"):
    from molfun.tracking import CometTracker
    trackers.append(CometTracker(
        project_name="molfun-kinase",
        experiment_name="lora-r8-pairformer",
    ))
    print("[+] Comet ML enabled")

# MLflow — local server or MLFLOW_TRACKING_URI
if os.getenv("MLFLOW_TRACKING_URI") or Path("mlruns").exists():
    from molfun.tracking import MLflowTracker
    trackers.append(MLflowTracker(
        experiment_name="molfun-kinase",
        run_name="lora-r8-pairformer",
    ))
    print("[+] MLflow enabled")

# HuggingFace Hub — set HF_TOKEN
if os.getenv("HF_TOKEN"):
    from molfun.tracking import HuggingFaceTracker
    trackers.append(HuggingFaceTracker(
        repo_id="your-username/kinase-lora",
    ))
    print("[+] Hugging Face Hub enabled")

tracker = CompositeTracker(trackers)

# ── 2. Log hyperparameters ───────────────────────────────────────────

hparams = {
    "model": "openfold",
    "strategy": "lora",
    "lora_rank": 8,
    "lora_alpha": 16.0,
    "lr_lora": 2e-4,
    "lr_head": 1e-3,
    "epochs": 20,
    "max_seq_len": 256,
    "attention": "flash",
    "n_blocks": 8,
    "structure_module": "ipa",
    "dataset": "PDBbind-kinases",
    "n_train": 150,
    "n_val": 25,
    "n_test": 25,
}

tracker.log_params(hparams)

# ── 3. Simulated training loop ───────────────────────────────────────

# In a real project you'd use strategy.fit(model, train_loader, val_loader, tracker=tracker)
# Here we simulate the loop to show what gets logged:

print("\nSimulating training loop...\n")

for epoch in range(1, 6):
    # Simulated metrics (replace with real training)
    train_loss = 2.0 / epoch + torch.randn(1).item() * 0.05
    val_loss = 2.2 / epoch + torch.randn(1).item() * 0.08
    val_mae = 1.5 / epoch + torch.randn(1).item() * 0.03
    lr = hparams["lr_lora"] * (0.95 ** epoch)

    tracker.log_metrics({
        "train/loss": train_loss,
        "val/loss": val_loss,
        "val/mae": val_mae,
        "lr": lr,
    }, step=epoch)

    print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_mae={val_mae:.4f}")

# ── 4. Log artifacts ─────────────────────────────────────────────────

# Save a dummy checkpoint to demonstrate artifact logging
checkpoint_dir = Path("runs/tracked_experiment/checkpoint")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
torch.save({"epoch": 5, "model_state": {}}, checkpoint_dir / "model.pt")

tracker.log_artifact(str(checkpoint_dir), name="best_checkpoint")

# ── 5. End tracking ──────────────────────────────────────────────────

tracker.end()

print("\nExperiment logged to all configured backends.")
print("Check your dashboards:")
print("  W&B:    https://wandb.ai/your-team/molfun-kinase")
print("  Comet:  https://www.comet.com/your-team/molfun-kinase")
print("  MLflow: http://localhost:5000")
print("  HF Hub: https://huggingface.co/your-username/kinase-lora")
