# Loss Functions

Molfun provides a registry of loss functions for structure prediction and
affinity tasks. All losses implement the `LossFunction` abstract base class
and are registered in `LOSS_REGISTRY`.

## Quick Start

```python
from molfun.losses import LOSS_REGISTRY, MSELoss, PearsonLoss

# Use the registry
loss_fn = LOSS_REGISTRY["mse"]()
result = loss_fn(preds, targets)
print(result)  # {"affinity_loss": tensor(0.42)}

# Or instantiate directly
loss_fn = PearsonLoss()
result = loss_fn(preds, targets)
# {"affinity_loss": tensor(0.15)}

# Register a custom loss
from molfun.losses import LossFunction

@LOSS_REGISTRY.register("tmscore")
class TMScoreLoss(LossFunction):
    def forward(self, preds, targets=None, batch=None):
        score = compute_tm_score(preds, targets)
        return {"tmscore_loss": 1.0 - score}
```

## LOSS_REGISTRY

::: molfun.losses.base.LOSS_REGISTRY
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## LossFunction (ABC)

::: molfun.losses.base.LossFunction
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Abstract base class for all loss functions. Subclasses must implement
`forward()`.

### Forward Signature

```python
def forward(
    self,
    preds: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    batch: Optional[dict] = None,
) -> dict[str, torch.Tensor]:
    ...
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `preds` | `Tensor` | Model predictions |
| `targets` | `Tensor \| None` | Ground truth labels |
| `batch` | `dict \| None` | Full feature dict from the DataLoader |

**Returns:** `dict[str, Tensor]` mapping loss term names to scalar tensors.

---

## Built-in Losses

### MSELoss

::: molfun.losses.affinity.MSELoss
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

Mean Squared Error loss for regression tasks.

```python
from molfun.losses import MSELoss

loss_fn = MSELoss()
result = loss_fn(preds, targets)
# {"affinity_loss": tensor(...)}
```

---

### MAELoss

::: molfun.losses.affinity.MAELoss
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

Mean Absolute Error (L1) loss.

```python
from molfun.losses import MAELoss

loss_fn = MAELoss()
result = loss_fn(preds, targets)
```

---

### HuberLoss

::: molfun.losses.affinity.HuberLoss
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

Huber (Smooth L1) loss -- less sensitive to outliers than MSE.

```python
from molfun.losses import HuberLoss

loss_fn = HuberLoss(delta=1.0)
result = loss_fn(preds, targets)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delta` | `float` | `1.0` | Threshold for switching between L1 and L2 |

---

### PearsonLoss

::: molfun.losses.affinity.PearsonLoss
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

1 minus Pearson correlation coefficient. Optimizes for ranking rather than
absolute accuracy.

```python
from molfun.losses import PearsonLoss

loss_fn = PearsonLoss()
result = loss_fn(preds, targets)
# {"affinity_loss": tensor(...)}  # 0 = perfect correlation
```

---

### OpenFoldLoss (FAPE)

::: molfun.backends.openfold.loss.OpenFoldLoss
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

Structure prediction loss from AlphaFold2/OpenFold, including FAPE
(Frame Aligned Point Error) and auxiliary losses.

```python
from molfun.losses import OpenFoldLoss

# Full loss (FAPE + aux)
loss_fn = OpenFoldLoss(config)
result = loss_fn(raw_outputs, batch=feature_dict)
# {"structure_loss": tensor(...), "fape": tensor(...), "aux": tensor(...)}

# FAPE only
loss_fn = OpenFoldLoss.fape_only(config)
```

---

## Combining Losses

```python
from molfun.losses import LOSS_REGISTRY

# Use multiple losses with weights
mse = LOSS_REGISTRY["mse"]()
pearson = LOSS_REGISTRY["pearson"]()

result_mse = mse(preds, targets)
result_pearson = pearson(preds, targets)

total = result_mse["affinity_loss"] + 0.5 * result_pearson["affinity_loss"]
total.backward()
```

## Registered Names

| Registry Key | Class | Task |
|-------------|-------|------|
| `"mse"` | `MSELoss` | Affinity regression |
| `"mae"` | `MAELoss` | Affinity regression |
| `"huber"` | `HuberLoss` | Affinity regression |
| `"pearson"` | `PearsonLoss` | Affinity ranking |
| `"openfold"` | `OpenFoldLoss` | Structure prediction |
