# MolfunStructureModel

The central facade for all Molfun operations: loading pretrained models,
running predictions, fine-tuning, module swapping, export, and Hub integration.

## Quick Start

```python
from molfun import MolfunStructureModel

# Load a pretrained model
model = MolfunStructureModel.from_pretrained("openfold_v2")

# Predict a structure
output = model.predict("MKFLILLFNILCLFPVLAADNH...")

# Fine-tune on a custom dataset
model.fit(
    train_dataset=train_ds,
    val_dataset=val_ds,
    epochs=10,
    strategy="lora",
)

# Save and push to hub
model.save("./my_model")
model.push_to_hub("myorg/finetuned-openfold")
```

## Class Reference

::: molfun.models.structure.MolfunStructureModel
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      heading_level: 3

## Class Methods

### from_pretrained

Load a model from a named pretrained checkpoint.

```python
model = MolfunStructureModel.from_pretrained(
    name="openfold_v2",
    device="cuda",
    dtype=torch.float16,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Pretrained model name (see `available_pretrained()`) |
| `device` | `str \| torch.device` | `"cpu"` | Target device |
| `dtype` | `torch.dtype` | `torch.float32` | Model precision |

**Returns:** `MolfunStructureModel`

---

### from_hub

Download and load a model from the Molfun Hub.

```python
model = MolfunStructureModel.from_hub("myorg/finetuned-openfold")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo_id` | `str` | *required* | Hub repository identifier |
| `revision` | `str \| None` | `None` | Specific revision / tag |
| `device` | `str \| torch.device` | `"cpu"` | Target device |

**Returns:** `MolfunStructureModel`

---

### from_custom

Build a model from modular components via `ModelBuilder`.

```python
model = MolfunStructureModel.from_custom(
    embedder="input",
    block="pairformer",
    n_blocks=48,
    structure_module="ipa",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedder` | `str` | *required* | Registered embedder name |
| `block` | `str` | *required* | Registered block name |
| `n_blocks` | `int` | *required* | Number of trunk blocks |
| `structure_module` | `str` | *required* | Registered structure module name |
| `**configs` | `dict` | `{}` | Extra configuration passed to each component |

**Returns:** `MolfunStructureModel`

---

## Instance Methods

### predict

Run inference on one or more sequences.

```python
output = model.predict(
    "MKFLILLFNILCLFPVLAADNH",
    num_recycles=3,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequence` | `str \| list[str]` | *required* | Amino acid sequence(s) |
| `num_recycles` | `int` | `3` | Number of recycling iterations |
| `msa` | `str \| None` | `None` | Path to MSA file (A3M) |

**Returns:** `TrunkOutput` with `.positions`, `.plddt`, `.pae` attributes.

---

### fit

Fine-tune the model using the selected strategy.

```python
model.fit(
    train_dataset=train_ds,
    val_dataset=val_ds,
    epochs=10,
    strategy="lora",
    lr=1e-4,
    batch_size=2,
    tracker="wandb",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_dataset` | `Dataset` | *required* | Training dataset |
| `val_dataset` | `Dataset \| None` | `None` | Validation dataset |
| `epochs` | `int` | `10` | Number of training epochs |
| `strategy` | `str` | `"full"` | Fine-tuning strategy: `"full"`, `"head_only"`, `"lora"`, `"partial"` |
| `lr` | `float` | `1e-4` | Learning rate |
| `batch_size` | `int` | `1` | Batch size |
| `tracker` | `str \| BaseTracker \| None` | `None` | Experiment tracker |
| `**kwargs` | `dict` | `{}` | Additional strategy-specific arguments |

**Returns:** Training metrics dict.

---

### forward

Low-level forward pass (used internally by `predict` and `fit`).

```python
output = model.forward(batch)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `batch` | `Batch \| dict` | Feature dictionary from a DataLoader |

**Returns:** `TrunkOutput`

---

### save / load

Persist and restore model checkpoints.

```python
model.save("./checkpoints/epoch_10")
model = MolfunStructureModel.load("./checkpoints/epoch_10")
```

---

### merge / unmerge

Merge or unmerge LoRA adapters into the base weights.

```python
model.merge()    # fuse adapters into base weights
model.unmerge()  # separate them back out
```

---

### swap / swap_all

Replace individual or all modules of a given type at runtime.

```python
# Swap a single attention module
model.swap("attention", "flash")

# Swap all blocks
model.swap_all("block", "pairformer")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `module_type` | `str` | Module category: `"attention"`, `"block"`, `"embedder"`, `"structure_module"` |
| `name` | `str` | Name of the registered replacement module |

---

### discover_modules

List all swappable modules currently in the model.

```python
modules = model.discover_modules()
# {"attention": ["layer_0", "layer_1", ...], "block": [...], ...}
```

**Returns:** `dict[str, list[str]]`

---

### push_to_hub

Upload the model to the Molfun Hub.

```python
model.push_to_hub("myorg/finetuned-openfold", private=True)
```

---

### export_onnx / export_torchscript

Export the model for deployment.

```python
model.export_onnx("model.onnx", opset_version=17)
model.export_torchscript("model.pt")
```

---

### example_dataset

Create a small synthetic dataset for testing.

```python
ds = MolfunStructureModel.example_dataset(n=100, task="structure")
```

---

## Static / Class Methods

### available_models

```python
MolfunStructureModel.available_models()
# ["openfold_v1", "openfold_v2", ...]
```

### available_heads

```python
MolfunStructureModel.available_heads()
# ["structure", "affinity", "property", ...]
```

### available_pretrained

```python
MolfunStructureModel.available_pretrained()
# ["openfold_v2", "openfold_finetuned_casp15", ...]
```

### summary

Print a summary of model architecture and parameter counts.

```python
model.summary()
```
