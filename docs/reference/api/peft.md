# PEFT -- Parameter-Efficient Fine-Tuning

The `molfun.training.peft` module provides utilities for applying
parameter-efficient fine-tuning methods (LoRA, IA3) to any Molfun model.

## Quick Start

```python
from molfun import MolfunStructureModel
from molfun.training.peft import MolfunPEFT

model = MolfunStructureModel.from_pretrained("openfold_v2")

# Apply LoRA adapters
peft = MolfunPEFT.lora(model, rank=8, alpha=16, target_modules=["q_proj", "v_proj"])

# Check trainable parameters
peft.summary()
# Total params: 93.2M | Trainable: 0.3M (0.32%)

# After training, merge adapters into base weights
peft.merge()

# Or save/load adapters separately
peft.save("./lora_adapters")
peft = MolfunPEFT.load("./lora_adapters", model)
```

## MolfunPEFT

::: molfun.training.peft.MolfunPEFT
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### lora (class method)

Apply LoRA (Low-Rank Adaptation) to the model.

```python
peft = MolfunPEFT.lora(
    model,
    rank=8,
    alpha=16,
    dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `MolfunStructureModel` | *required* | Model to adapt |
| `rank` | `int` | `8` | LoRA rank (lower = fewer parameters) |
| `alpha` | `float` | `16.0` | LoRA scaling factor |
| `dropout` | `float` | `0.0` | Dropout applied to LoRA layers |
| `target_modules` | `list[str] \| None` | `None` | Module name patterns to target. `None` targets all linear layers. |

**Returns:** `MolfunPEFT`

---

### ia3 (class method)

Apply IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations).

```python
peft = MolfunPEFT.ia3(
    model,
    target_modules=["k_proj", "v_proj", "down_proj"],
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `MolfunStructureModel` | *required* | Model to adapt |
| `target_modules` | `list[str] \| None` | `None` | Module name patterns to target |

**Returns:** `MolfunPEFT`

---

### apply

Apply the PEFT configuration to the model. Called automatically by `lora()` and `ia3()`.

```python
peft.apply()
```

---

### trainable_parameters

Return an iterator over only the trainable (adapter) parameters.

```python
for name, param in peft.trainable_parameters():
    print(f"{name}: {param.shape}")
```

**Returns:** `Iterator[tuple[str, Parameter]]`

---

### merge

Merge adapter weights into the base model weights.

```python
peft.merge()
```

After merging, the model behaves as a standard model with no adapter
overhead at inference time.

---

### unmerge

Reverse a previous `merge()`, restoring the separate adapter weights.

```python
peft.unmerge()
```

---

### save

Save adapter weights to disk (without the base model).

```python
peft.save("./lora_adapters")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Directory to save adapter weights and config |

---

### load (class method)

Load adapter weights from disk and apply them to a model.

```python
peft = MolfunPEFT.load("./lora_adapters", model)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Directory containing adapter weights |
| `model` | `MolfunStructureModel` | Base model to attach adapters to |

**Returns:** `MolfunPEFT`

---

### summary

Print a summary of total vs trainable parameters.

```python
peft.summary()
# Total parameters:     93,215,488
# Trainable parameters:    294,912 (0.32%)
# PEFT method: LoRA (rank=8, alpha=16.0)
```

---

## LoRALinear

::: molfun.training.peft.LoRALinear
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Low-rank adapter layer that wraps a standard `nn.Linear`.

```python
from molfun.training.peft import LoRALinear

# Wrap an existing linear layer
lora_layer = LoRALinear(
    original=model.trunk.layers[0].attention.q_proj,
    rank=8,
    alpha=16.0,
    dropout=0.05,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `original` | `nn.Linear` | *required* | The linear layer to wrap |
| `rank` | `int` | `8` | Rank of the low-rank decomposition |
| `alpha` | `float` | `16.0` | Scaling factor (effective scale = alpha / rank) |
| `dropout` | `float` | `0.0` | Dropout probability on the LoRA path |

The forward pass computes: `output = original(x) + (dropout(x) @ A^T @ B^T) * (alpha / rank)`

Where `A` is shape `(rank, in_features)` and `B` is shape `(out_features, rank)`.
