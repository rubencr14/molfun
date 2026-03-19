# ModuleSwapper

Hot-swap modular components in a live model without rebuilding from scratch.
Useful for architecture search, ablation studies, and runtime optimization.

## Quick Start

```python
from molfun.modules.swapper import ModuleSwapper
from molfun import MolfunStructureModel

model = MolfunStructureModel.from_pretrained("openfold_v2")

# Discover swappable modules
swappable = ModuleSwapper.discover(model)
print(swappable)
# {"attention": ["layer_0.attn", ...], "block": ["layer_0", ...], ...}

# Swap a single module
ModuleSwapper.swap(model, module_type="attention", name="flash")

# Swap all modules of a type
ModuleSwapper.swap_all(model, module_type="attention", name="linear")

# Swap by type (swap all instances of a specific class)
ModuleSwapper.swap_by_type(
    model,
    old_type="StandardAttention",
    new_type="flash",
)
```

## Class Reference

::: molfun.modules.swapper.ModuleSwapper
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### swap

Replace a specific module instance with a registered alternative.

```python
ModuleSwapper.swap(
    model,
    module_type="attention",
    name="flash",
    module_path="trunk.layers.0.attn",  # optional: target a specific submodule
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | *required* | Model to modify |
| `module_type` | `str` | *required* | Category: `"attention"`, `"block"`, `"embedder"`, `"structure_module"` |
| `name` | `str` | *required* | Registered replacement name |
| `module_path` | `str \| None` | `None` | Dotted path to a specific submodule to replace |
| `**kwargs` | `dict` | `{}` | Extra args passed to the new module constructor |

---

### swap_all

Replace all modules of a given type throughout the model.

```python
ModuleSwapper.swap_all(model, module_type="attention", name="flash")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | *required* | Model to modify |
| `module_type` | `str` | *required* | Category to replace |
| `name` | `str` | *required* | Registered replacement name |
| `**kwargs` | `dict` | `{}` | Extra args for the new modules |

---

### swap_by_type

Replace all instances of a specific class (regardless of registry category).

```python
ModuleSwapper.swap_by_type(
    model,
    old_type="StandardAttention",
    new_type="flash",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | *required* | Model to modify |
| `old_type` | `str \| type` | *required* | Class name or class to replace |
| `new_type` | `str` | *required* | Registered replacement name |
| `**kwargs` | `dict` | `{}` | Extra args for the new modules |

---

### discover

List all swappable modules found in the model, grouped by type.

```python
swappable = ModuleSwapper.discover(model)
# {
#     "attention": ["trunk.layers.0.attn", "trunk.layers.1.attn", ...],
#     "block": ["trunk.layers.0", "trunk.layers.1", ...],
#     "embedder": ["embedder"],
#     "structure_module": ["structure_module"],
# }
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | Model to inspect |

**Returns:** `dict[str, list[str]]` mapping module types to lists of dotted paths.

---

## Example: Architecture Ablation

```python
from molfun import MolfunStructureModel
from molfun.modules.swapper import ModuleSwapper

model = MolfunStructureModel.from_pretrained("openfold_v2")

results = {}
for attn_type in ["standard", "flash", "linear", "gated"]:
    ModuleSwapper.swap_all(model, "attention", attn_type)
    output = model.predict("MKFLILLFNILCLFPVLAADNH...")
    results[attn_type] = output.plddt.mean().item()

print(results)
```
