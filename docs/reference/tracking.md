# Experiment Tracking

Molfun integrates with popular experiment tracking platforms through a
unified `BaseTracker` interface. Multiple trackers can be composed to log
to several platforms simultaneously.

## Quick Start

```python
from molfun import MolfunStructureModel

model = MolfunStructureModel.from_pretrained("openfold_v2")

# Pass tracker name to fit()
model.fit(train_dataset=ds, strategy="lora", tracker="wandb", epochs=10)

# Or instantiate a tracker directly
from molfun.tracking import WandbTracker

tracker = WandbTracker(project="molfun-experiments", name="lora-r8")
model.fit(train_dataset=ds, strategy="lora", tracker=tracker, epochs=10)

# Compose multiple trackers
from molfun.tracking import CompositeTracker, WandbTracker, ConsoleTracker

tracker = CompositeTracker([
    WandbTracker(project="molfun"),
    ConsoleTracker(),
])
```

## BaseTracker

::: molfun.tracking.base.BaseTracker
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Abstract base class for all experiment trackers. Implement this to add
support for a new tracking platform.

### Required Methods

| Method | Description |
|--------|-------------|
| `start_run(**kwargs)` | Initialize a tracking run |
| `log_metrics(metrics, step)` | Log a dict of scalar metrics |
| `log_config(config)` | Log experiment configuration/hyperparameters |
| `log_artifact(path, name)` | Log a file artifact (model checkpoint, etc.) |
| `log_text(text, name)` | Log a text blob (predictions, summaries) |
| `end_run()` | Finalize and close the run |

```python
from molfun.tracking.base import BaseTracker

class MyTracker(BaseTracker):
    def start_run(self, **kwargs):
        ...
    def log_metrics(self, metrics: dict, step: int | None = None):
        ...
    def log_config(self, config: dict):
        ...
    def log_artifact(self, path: str, name: str | None = None):
        ...
    def log_text(self, text: str, name: str | None = None):
        ...
    def end_run(self):
        ...
```

---

## WandbTracker

::: molfun.tracking.wandb_tracker.WandbTracker
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Weights & Biases experiment tracker.

```python
from molfun.tracking import WandbTracker

tracker = WandbTracker(
    project="molfun-structure",
    name="lora-rank8-run1",
    entity="my-team",
    tags=["lora", "openfold"],
)

tracker.start_run()
tracker.log_config({"rank": 8, "lr": 1e-4})
tracker.log_metrics({"loss": 0.42, "rmsd": 1.5}, step=100)
tracker.log_artifact("./checkpoint.pt", name="best_model")
tracker.end_run()
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | `str` | `"molfun"` | W&B project name |
| `name` | `str \| None` | `None` | Run name |
| `entity` | `str \| None` | `None` | W&B team/entity |
| `tags` | `list[str]` | `[]` | Run tags |

---

## CometTracker

::: molfun.tracking.comet_tracker.CometTracker
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Comet ML experiment tracker.

```python
from molfun.tracking import CometTracker

tracker = CometTracker(
    project_name="molfun-experiments",
    api_key="...",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_name` | `str` | *required* | Comet project name |
| `api_key` | `str \| None` | `None` | API key (reads `COMET_API_KEY` env var if `None`) |
| `workspace` | `str \| None` | `None` | Comet workspace |

---

## MLflowTracker

::: molfun.tracking.mlflow_tracker.MLflowTracker
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

MLflow experiment tracker.

```python
from molfun.tracking import MLflowTracker

tracker = MLflowTracker(
    experiment_name="molfun-structure",
    tracking_uri="http://localhost:5000",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | `str` | `"molfun"` | MLflow experiment name |
| `tracking_uri` | `str \| None` | `None` | MLflow tracking server URI |

---

## LangfuseTracker

::: molfun.tracking.langfuse_tracker.LangfuseTracker
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Langfuse tracker for monitoring AI/ML pipelines.

```python
from molfun.tracking import LangfuseTracker

tracker = LangfuseTracker(
    public_key="...",
    secret_key="...",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `public_key` | `str \| None` | `None` | Langfuse public key (or `LANGFUSE_PUBLIC_KEY` env var) |
| `secret_key` | `str \| None` | `None` | Langfuse secret key (or `LANGFUSE_SECRET_KEY` env var) |
| `host` | `str` | `"https://cloud.langfuse.com"` | Langfuse host URL |

---

## HuggingFaceTracker

::: molfun.tracking.hf_tracker.HuggingFaceTracker
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

HuggingFace Hub tracker that logs metrics and model cards.

```python
from molfun.tracking import HuggingFaceTracker

tracker = HuggingFaceTracker(repo_id="myorg/molfun-model")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo_id` | `str` | *required* | HuggingFace Hub repository ID |
| `token` | `str \| None` | `None` | HF API token |

---

## ConsoleTracker

::: molfun.tracking.console.ConsoleTracker
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Simple tracker that prints metrics to the console. Useful for debugging
and local development.

```python
from molfun.tracking import ConsoleTracker

tracker = ConsoleTracker(print_every=10)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `print_every` | `int` | `1` | Print metrics every N steps |

---

## CompositeTracker

::: molfun.tracking.composite.CompositeTracker
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Compose multiple trackers to log to several platforms simultaneously.

```python
from molfun.tracking import CompositeTracker, WandbTracker, ConsoleTracker, MLflowTracker

tracker = CompositeTracker([
    WandbTracker(project="molfun"),
    MLflowTracker(experiment_name="molfun"),
    ConsoleTracker(),
])

# All calls are forwarded to every child tracker
tracker.start_run()
tracker.log_metrics({"loss": 0.42}, step=1)
tracker.end_run()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `trackers` | `list[BaseTracker]` | List of tracker instances to compose |
