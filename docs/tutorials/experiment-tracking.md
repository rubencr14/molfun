---
title: Experiment Tracking
---

# Experiment Tracking

<span class="badge badge-beginner">Beginner</span> &nbsp; ~15 min

Track metrics, hyperparameters, and artifacts across training runs with Molfun's built-in
integrations for **Weights & Biases**, **Comet ML**, and **MLflow**. Use `CompositeTracker`
to log to multiple backends simultaneously.

---

## What You Will Learn

- Set up WandB, Comet, and MLflow trackers
- Pass a tracker to `model.fit()`
- Use `CompositeTracker` for multi-backend logging
- Log custom metrics, configs, and artifacts
- Best practices for experiment organization

---

## Quick Start

Adding tracking to any Molfun training run is a single extra argument:

```python
from molfun import MolfunStructureModel
from molfun.tracking import WandbTracker

model = MolfunStructureModel.from_pretrained("openfold_v1", device="cuda")

tracker = WandbTracker(project="my-protein-project")  # (1)!

model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    strategy=strategy,
    epochs=20,
    tracker=tracker,         # Just add this line
)
```

1. The tracker is initialized here but the run starts automatically when `fit()` is
   called. It ends when training completes.

---

## Tracker Setup

=== "Weights & Biases"

    Install and authenticate:

    ```bash
    pip install wandb
    wandb login
    ```

    ```python
    from molfun.tracking import WandbTracker

    tracker = WandbTracker(
        project="protein-stability",   # WandB project name
    )
    ```

    !!! tip "WandB features"

        WandB automatically logs:

        - Training and validation loss curves
        - Learning rate schedule
        - System metrics (GPU utilization, memory)
        - Model architecture summary

=== "Comet ML"

    Install and set your API key:

    ```bash
    pip install comet-ml
    export COMET_API_KEY="your-api-key"
    ```

    ```python
    from molfun.tracking import CometTracker

    tracker = CometTracker(
        project="protein-stability",   # Comet project name
    )
    ```

=== "MLflow"

    Install and optionally set a tracking URI:

    ```bash
    pip install mlflow
    ```

    ```python
    from molfun.tracking import MLflowTracker

    tracker = MLflowTracker(
        experiment="protein-stability",  # MLflow experiment name
    )
    ```

    !!! info "MLflow tracking server"

        By default, MLflow logs to a local `./mlruns` directory. To use a remote
        tracking server:

        ```python
        import mlflow
        mlflow.set_tracking_uri("http://your-mlflow-server:5000")

        tracker = MLflowTracker(experiment="protein-stability")
        ```

---

## CompositeTracker

Log to multiple backends simultaneously. Useful for teams where different members
prefer different tools.

```python
from molfun.tracking import WandbTracker, CometTracker, CompositeTracker

tracker = CompositeTracker([
    WandbTracker(project="protein-stability"),
    CometTracker(project="protein-stability"),
])

# Use exactly like a single tracker
model.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    strategy=strategy,
    epochs=20,
    tracker=tracker,
)
```

All methods (`log_metrics`, `log_config`, `log_artifact`) are forwarded to every
backend.

---

## Manual Tracking API

Beyond automatic logging during `fit()`, you can use the tracker API directly for
custom metrics and artifacts.

### Log Metrics

```python
tracker.start_run(run_name="lora-rank8-experiment")

# Log scalar metrics
tracker.log_metrics({
    "pearson_r": 0.85,
    "rmse": 1.23,
    "best_epoch": 12,
}, step=0)

# Log metrics at specific steps
for epoch in range(20):
    tracker.log_metrics({"custom/my_metric": compute_metric()}, step=epoch)
```

### Log Configuration

```python
tracker.log_config({
    "model": "openfold_v1",
    "strategy": "lora",
    "rank": 8,
    "alpha": 16.0,
    "lr_lora": 1e-4,
    "lr_head": 1e-3,
    "batch_size": 8,
    "max_seq_length": 512,
    "dataset_size": len(train_ds),
})
```

### Log Artifacts

```python
# Log a saved model
tracker.log_artifact("models/affinity_lora", name="trained-model")

# Log a plot
tracker.log_artifact("results/scatter_plot.png", name="evaluation-plot")

# Log text (predictions, notes)
tracker.log_text("Best run achieved r=0.85 with rank=8, alpha=16")
```

### End a Run

```python
tracker.end_run()
```

---

## Comparing Runs

A common workflow is to sweep over hyperparameters and compare results.

```python
from molfun import MolfunStructureModel
from molfun.training import LoRAFinetune
from molfun.tracking import WandbTracker

ranks = [4, 8, 16]
alphas = [8.0, 16.0, 32.0]

for rank in ranks:
    for alpha in alphas:
        tracker = WandbTracker(project="lora-sweep")
        tracker.start_run(run_name=f"rank{rank}-alpha{alpha}")

        tracker.log_config({
            "rank": rank,
            "alpha": alpha,
        })

        model = MolfunStructureModel.from_pretrained(
            "openfold_v1", device="cuda",
            head="affinity",
            head_config={"hidden_dim": 256, "num_layers": 2},
        )

        strategy = LoRAFinetune(
            rank=rank, alpha=alpha,
            lr_lora=1e-4, lr_head=1e-3,
        )

        model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            strategy=strategy,
            epochs=15,
            tracker=tracker,
        )

        # Log final metrics
        final_metrics = evaluate(model, test_loader)
        tracker.log_metrics(final_metrics)
        tracker.end_run()
```

---

## Best Practices

!!! success "Naming conventions"

    Use consistent, descriptive run names:

    ```
    {strategy}-{key_param}-{dataset}-{date}
    ```

    Examples: `lora-r8-pdbbind-20260319`, `headonly-stability-v2`

!!! success "Log everything reproducible"

    Always log:

    - [x] All hyperparameters (strategy params, LR, batch size)
    - [x] Dataset metadata (size, split ratios, random seed)
    - [x] Model configuration (head type, hidden dims)
    - [x] Environment info (GPU type, PyTorch version)

!!! success "Project organization"

    - **One project per task**: `stability-prediction`, `binding-affinity`, etc.
    - **Tags for grouping**: Tag runs by strategy, dataset version, or experiment phase
    - **Artifact versioning**: Log model checkpoints as artifacts for reproducibility

!!! warning "Avoid"

    - Logging too frequently (every batch) on large datasets --- log every N steps instead
    - Forgetting to call `end_run()` when using the manual API
    - Mixing unrelated experiments in the same project

---

## Next Steps

- **Run your first tracked experiment**: Try [Stability Prediction](stability-prediction.md)
  with a tracker enabled.
- **Automate tracking in pipelines**: See [YAML Pipelines](yaml-pipelines.md) where
  tracking is configured declaratively.
- **Sweep hyperparameters**: Combine tracking with the comparison pattern above for
  systematic experiments.
