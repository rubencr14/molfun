# Fine-Tuning Strategies

Molfun provides four fine-tuning strategies, all implementing the
`FinetuneStrategy` abstract base class. Each strategy defines which
parameters are trainable and how they are grouped for the optimizer.

## Quick Start

```python
from molfun import MolfunStructureModel

model = MolfunStructureModel.from_pretrained("openfold_v2")

# Via the model facade
model.fit(train_dataset=ds, strategy="lora", epochs=5)

# Or instantiate directly
from molfun.training.lora import LoRAFinetune

strategy = LoRAFinetune(rank=8, alpha=16)
strategy.fit(model, train_dataset=ds, epochs=5)
```

## Base Class

::: molfun.training.base.FinetuneStrategy
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

The `fit()` template method implements the training loop. Subclasses
customize behaviour through `_setup_impl()` and `param_groups()`.

---

## FullFinetune

::: molfun.training.full.FullFinetune
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Fine-tune all model parameters. Highest capacity but requires the most
memory and data.

```python
from molfun.training.full import FullFinetune

strategy = FullFinetune(lr=1e-4, weight_decay=0.01)
strategy.fit(model, train_dataset=ds, val_dataset=val_ds, epochs=20)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | `1e-4` | Learning rate |
| `weight_decay` | `float` | `0.0` | L2 regularization |
| `max_grad_norm` | `float \| None` | `1.0` | Gradient clipping norm |
| `warmup_steps` | `int` | `0` | Linear warmup steps |
| `scheduler` | `str` | `"cosine"` | LR scheduler type |

---

## HeadOnlyFinetune

::: molfun.training.head_only.HeadOnlyFinetune
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Freeze the trunk and fine-tune only the prediction head. Fast and
memory-efficient; best for transfer learning with limited data.

```python
from molfun.training.head_only import HeadOnlyFinetune

strategy = HeadOnlyFinetune(lr=5e-4)
strategy.fit(model, train_dataset=ds, epochs=50)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | `float` | `5e-4` | Learning rate |
| `weight_decay` | `float` | `0.0` | L2 regularization |
| `max_grad_norm` | `float \| None` | `1.0` | Gradient clipping norm |

---

## LoRAFinetune

::: molfun.training.lora.LoRAFinetune
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Apply LoRA adapters and fine-tune only the adapter parameters.
Excellent trade-off between capacity and efficiency.

```python
from molfun.training.lora import LoRAFinetune

strategy = LoRAFinetune(
    rank=8,
    alpha=16,
    target_modules=["q_proj", "v_proj"],
    lr=2e-4,
)
strategy.fit(model, train_dataset=ds, epochs=10)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rank` | `int` | `8` | LoRA rank |
| `alpha` | `float` | `16.0` | LoRA scaling factor |
| `dropout` | `float` | `0.0` | LoRA dropout |
| `target_modules` | `list[str] \| None` | `None` | Module name patterns to adapt |
| `lr` | `float` | `2e-4` | Learning rate |
| `weight_decay` | `float` | `0.0` | L2 regularization |
| `max_grad_norm` | `float \| None` | `1.0` | Gradient clipping norm |

---

## PartialFinetune

::: molfun.training.partial.PartialFinetune
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Unfreeze the last N trunk blocks plus the prediction head. A middle ground
between full fine-tuning and head-only.

```python
from molfun.training.partial import PartialFinetune

strategy = PartialFinetune(
    n_unfrozen_blocks=4,
    lr=1e-4,
    head_lr=5e-4,
)
strategy.fit(model, train_dataset=ds, epochs=15)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_unfrozen_blocks` | `int` | `4` | Number of trunk blocks to unfreeze (from the end) |
| `lr` | `float` | `1e-4` | Learning rate for unfrozen trunk blocks |
| `head_lr` | `float \| None` | `None` | Separate learning rate for the head (defaults to `lr`) |
| `weight_decay` | `float` | `0.0` | L2 regularization |
| `max_grad_norm` | `float \| None` | `1.0` | Gradient clipping norm |

---

## Comparison

| Strategy | Trainable Params | Memory | Best For |
|----------|-----------------|--------|----------|
| **FullFinetune** | 100% | High | Large datasets, maximum accuracy |
| **HeadOnlyFinetune** | ~1-2% | Low | Small datasets, fast iteration |
| **LoRAFinetune** | ~0.3-1% | Low | General-purpose fine-tuning |
| **PartialFinetune** | ~10-30% | Medium | Moderate data, domain adaptation |
