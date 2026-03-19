---
title: YAML Pipelines
---

# YAML Pipelines

<span class="badge badge-intermediate">Intermediate</span> &nbsp; ~20 min

Define reproducible, end-to-end workflows with `Pipeline.from_yaml()`. A single YAML file
describes every step --- from data fetching through preprocessing, training, and evaluation
--- making experiments easy to share, version, and rerun.

---

## What You Will Learn

- Write YAML pipeline recipes with Molfun's pipeline format
- Use built-in steps: `fetch`, `preprocess`, `train`, `evaluate`
- Run pipelines from Python or the CLI
- Write custom pipeline steps
- Organize recipes for common tasks

---

## Pipeline Architecture

```mermaid
graph LR
    YAML["recipe.yaml"] --> PARSE["Pipeline.from_yaml()"]
    PARSE --> S1["Step 1: fetch"]
    S1 --> S2["Step 2: preprocess"]
    S2 --> S3["Step 3: train"]
    S3 --> S4["Step 4: evaluate"]
    S4 --> OUT["Results + Artifacts"]

    style YAML fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style PARSE fill:#7c3aed,stroke:#6d28d9,color:#ffffff
    style S1 fill:#0d9488,stroke:#0f766e,color:#ffffff
    style S2 fill:#d97706,stroke:#b45309,color:#ffffff
    style S3 fill:#c026d3,stroke:#a21caf,color:#ffffff
    style S4 fill:#16a34a,stroke:#15803d,color:#ffffff
    style OUT fill:#0891b2,stroke:#0e7490,color:#ffffff
```

---

## Step 1: Your First Pipeline

Create a file called `stability_pipeline.yaml`:

```yaml title="stability_pipeline.yaml"
name: stability-prediction
description: Train a stability predictor with HeadOnly strategy

# ── Data ──────────────────────────────────────────────────
fetch:
  type: csv
  path: data/stability_data.csv
  columns:
    sequence: sequence
    label: ddg

preprocess:
  max_length: 512
  split:
    test_size: 0.2
    random_state: 42
  loader:
    batch_size: 4
    shuffle: true

# ── Model ─────────────────────────────────────────────────
model:
  pretrained: openfold_v1
  device: cuda
  head: affinity
  head_config:
    hidden_dim: 256
    num_layers: 2
    dropout: 0.1

# ── Training ──────────────────────────────────────────────
train:
  strategy: head_only
  strategy_config:
    lr: 1.0e-3
    weight_decay: 1.0e-4
  epochs: 20
  checkpoint_dir: checkpoints/stability

# ── Evaluation ────────────────────────────────────────────
evaluate:
  metrics:
    - pearson
    - rmse
    - mae
  save_predictions: results/stability_predictions.csv

# ── Output ────────────────────────────────────────────────
output:
  save_model: models/stability_headonly
  # push_to_hub: your-username/stability-predictor  # Uncomment to push
```

---

## Step 2: Run the Pipeline

=== "Python"

    ```python
    from molfun.pipelines import Pipeline

    pipeline = Pipeline.from_yaml("stability_pipeline.yaml")
    results = pipeline.run()

    print(f"Pearson r: {results.metrics['pearson']:.4f}")
    print(f"RMSE:      {results.metrics['rmse']:.4f}")
    print(f"Model saved to: {results.model_path}")
    ```

=== "CLI"

    ```bash
    molfun pipeline run stability_pipeline.yaml
    ```

    ??? example "CLI output"

        ```
        [Pipeline] stability-prediction
        [Step 1/4] Fetching data from data/stability_data.csv...
          Loaded 200 samples
        [Step 2/4] Preprocessing...
          Train: 160 | Val: 40
        [Step 3/4] Training (HeadOnly, 20 epochs)...
          Epoch  1/20 | Train: 4.21 | Val: 3.89
          Epoch 10/20 | Train: 0.82 | Val: 0.94
          Epoch 20/20 | Train: 0.41 | Val: 0.52
        [Step 4/4] Evaluating...
          Pearson r: 0.8432
          RMSE:      0.7821
        [Done] Model saved to models/stability_headonly
        ```

---

## Step 3: Recipe Examples

### Binding Affinity with LoRA

```yaml title="affinity_lora.yaml"
name: affinity-lora
description: Binding affinity prediction with LoRA fine-tuning

fetch:
  type: pdbbind
  subset: refined
  max_samples: 500

preprocess:
  max_length: 512
  split:
    val_size: 0.1
    test_size: 0.1
    random_state: 42
  loader:
    batch_size: 8
    shuffle: true

model:
  pretrained: openfold_v1
  device: cuda
  head: affinity
  head_config:
    hidden_dim: 256
    num_layers: 2
    dropout: 0.1

train:
  strategy: lora
  strategy_config:
    rank: 8
    alpha: 16.0
    lr_lora: 1.0e-4
    lr_head: 1.0e-3
    warmup_steps: 100
    ema_decay: 0.999
  epochs: 15
  checkpoint_dir: checkpoints/affinity_lora

evaluate:
  metrics:
    - pearson
    - rmse
  save_predictions: results/affinity_predictions.csv

tracking:
  backend: wandb
  project: affinity-experiments

output:
  save_model: models/affinity_lora
  merge_lora: true
```

### Kinase Refinement with PartialFinetune

```yaml title="kinase_refinement.yaml"
name: kinase-refinement
description: Refine kinase structures with partial fine-tuning

fetch:
  type: collection
  name: kinases_human

preprocess:
  max_length: 600
  split:
    test_size: 0.2
    random_state: 42
  loader:
    batch_size: 2
    shuffle: true

model:
  pretrained: openfold_v1
  device: cuda

train:
  strategy: partial
  strategy_config:
    n_unfrozen_blocks: 4
    lr: 5.0e-5
  epochs: 25
  checkpoint_dir: checkpoints/kinase

evaluate:
  metrics:
    - fape
    - lddt
  save_predictions: results/kinase_structures/

tracking:
  backend: mlflow
  experiment: kinase-refinement

output:
  save_model: models/kinase_refined
```

---

## Step 4: Adding Tracking

Add a `tracking` section to any recipe to enable experiment tracking:

```yaml
tracking:
  backend: wandb               # wandb, comet, or mlflow
  project: my-project          # Project / experiment name
```

For multiple backends, use a list:

```yaml
tracking:
  backends:
    - type: wandb
      project: my-project
    - type: mlflow
      experiment: my-project
```

This automatically creates a `CompositeTracker` under the hood.

---

## Step 5: Custom Pipeline Steps

You can extend the pipeline with custom steps by writing a Python function and
referencing it in the YAML.

### Define a Custom Step

```python title="custom_steps.py"
from molfun.pipelines import register_step

@register_step("filter_by_length")
def filter_by_length(data, min_length=50, max_length=500):
    """Filter sequences by length."""
    filtered = [
        record for record in data
        if min_length <= len(record.sequence) <= max_length
    ]
    print(f"Filtered: {len(data)} -> {len(filtered)} sequences")
    return filtered
```

### Use in YAML

```yaml title="pipeline_with_custom_step.yaml"
name: filtered-stability
description: Stability prediction with length filtering

custom_steps:
  - custom_steps.py              # Load custom step definitions

fetch:
  type: csv
  path: data/stability_data.csv
  columns:
    sequence: sequence
    label: ddg

# Custom step inserted between fetch and preprocess
filter_by_length:
  min_length: 50
  max_length: 400

preprocess:
  max_length: 400
  split:
    test_size: 0.2
    random_state: 42
  loader:
    batch_size: 4
    shuffle: true

model:
  pretrained: openfold_v1
  device: cuda
  head: affinity
  head_config:
    hidden_dim: 256
    num_layers: 2

train:
  strategy: head_only
  strategy_config:
    lr: 1.0e-3
  epochs: 20

evaluate:
  metrics:
    - pearson
    - rmse

output:
  save_model: models/filtered_stability
```

---

## Step 6: Pipeline Composition

Run multiple pipelines in sequence, passing outputs between them:

```python
from molfun.pipelines import Pipeline

# Train pipeline
train_pipeline = Pipeline.from_yaml("train_recipe.yaml")
train_results = train_pipeline.run()

# Evaluation pipeline (uses the model from training)
eval_recipe = {
    "name": "evaluation",
    "model": {"path": train_results.model_path},
    "fetch": {"type": "csv", "path": "data/test_set.csv"},
    "evaluate": {"metrics": ["pearson", "rmse", "mae"]},
}
eval_pipeline = Pipeline.from_dict(eval_recipe)
eval_results = eval_pipeline.run()
```

---

## YAML Reference

??? info "Complete YAML schema"

    | Section | Key | Type | Description |
    |---------|-----|------|-------------|
    | **Top-level** | `name` | string | Pipeline name |
    | | `description` | string | Human-readable description |
    | | `custom_steps` | list[string] | Python files to load custom steps from |
    | **fetch** | `type` | string | `csv`, `pdbbind`, `collection`, `pdb_ids` |
    | | `path` | string | Path to CSV file (for `csv` type) |
    | | `subset` | string | PDBbind subset: `refined` or `general` |
    | | `name` | string | Collection name (for `collection` type) |
    | | `pdb_ids` | list[string] | PDB IDs to fetch (for `pdb_ids` type) |
    | | `max_samples` | int | Maximum number of samples |
    | **preprocess** | `max_length` | int | Maximum sequence length |
    | | `split.test_size` | float | Test split fraction |
    | | `split.val_size` | float | Validation split fraction |
    | | `split.random_state` | int | Random seed |
    | | `loader.batch_size` | int | DataLoader batch size |
    | | `loader.shuffle` | bool | Shuffle training data |
    | **model** | `pretrained` | string | Pretrained model name |
    | | `device` | string | `cuda` or `cpu` |
    | | `head` | string | Prediction head type |
    | | `head_config` | dict | Head configuration |
    | **train** | `strategy` | string | `full`, `head_only`, `lora`, `partial` |
    | | `strategy_config` | dict | Strategy-specific parameters |
    | | `epochs` | int | Number of training epochs |
    | | `checkpoint_dir` | string | Directory for checkpoints |
    | **evaluate** | `metrics` | list[string] | Metric names: `pearson`, `rmse`, `mae`, `fape`, `lddt` |
    | | `save_predictions` | string | Path to save predictions |
    | **tracking** | `backend` | string | `wandb`, `comet`, `mlflow` |
    | | `project` | string | Project name |
    | | `experiment` | string | Experiment name (MLflow) |
    | **output** | `save_model` | string | Path to save trained model |
    | | `merge_lora` | bool | Merge LoRA weights before saving |
    | | `push_to_hub` | string | HuggingFace Hub repo ID |

---

## Best Practices

!!! success "Version your recipes"

    Store YAML recipes in version control alongside your code. Each recipe is a
    complete, reproducible experiment definition.

!!! success "Use environment variables for paths"

    ```yaml
    fetch:
      type: csv
      path: ${DATA_DIR}/stability_data.csv

    output:
      save_model: ${MODEL_DIR}/stability_headonly
    ```

!!! success "Start simple, add complexity"

    Begin with the minimal recipe (fetch + model + train) and add preprocessing,
    evaluation, and tracking as needed.

---

## Next Steps

- **First time?** Start with the [Stability Prediction](stability-prediction.md) tutorial
  and convert it to a YAML pipeline.
- **Need custom steps?** The custom step API supports any Python function.
- **Team workflows?** Combine with [Experiment Tracking](experiment-tracking.md) to share
  results across team members.
