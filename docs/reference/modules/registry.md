# ModuleRegistry

A generic, type-safe plugin system used throughout Molfun for registering and
retrieving modular components (attention mechanisms, blocks, embedders,
structure modules, losses).

## Quick Start

```python
from molfun.modules.registry import ModuleRegistry

# Create a registry for a new component type
MY_REGISTRY = ModuleRegistry("my_component")

# Register a class
@MY_REGISTRY.register("custom")
class MyCustomComponent:
    def __init__(self, dim: int = 64):
        self.dim = dim

# Retrieve and instantiate
cls = MY_REGISTRY.get("custom")
instance = MY_REGISTRY.build("custom", dim=128)

# List all registered
print(MY_REGISTRY.list())  # ["custom"]
```

## Class Reference

::: molfun.modules.registry.ModuleRegistry
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### register

Decorator to register a class under a given name.

```python
@MY_REGISTRY.register("custom")
class MyComponent:
    ...
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique identifier for this component |

**Returns:** Decorator that registers the class and returns it unchanged.

**Raises:** `ValueError` if the name is already registered.

---

### build

Instantiate a registered class with the given arguments.

```python
instance = MY_REGISTRY.build("custom", dim=128, dropout=0.1)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Registered component name |
| `**kwargs` | `dict` | Arguments passed to the class constructor |

**Returns:** Instance of the registered class.

**Raises:** `KeyError` if the name is not registered.

---

### get

Retrieve the class (not an instance) by name.

```python
cls = MY_REGISTRY.get("custom")
instance = cls(dim=128)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Registered component name |

**Returns:** The registered class.

---

### list

Return all registered names as a sorted list of strings.

```python
names = MY_REGISTRY.list()
# ["custom", "default", "flash"]
```

**Returns:** `list[str]`

---

### Dunder Methods

```python
# Dictionary-style access
cls = MY_REGISTRY["custom"]

# Membership test
if "custom" in MY_REGISTRY:
    ...

# Iteration
for name in MY_REGISTRY:
    print(name)

# Length
print(len(MY_REGISTRY))  # number of registered components
```

## Built-in Registries

Molfun ships with several pre-populated registries:

| Registry | Module | Contents |
|----------|--------|----------|
| `ATTENTION_REGISTRY` | `molfun.modules.attention` | flash, standard, linear, gated |
| `BLOCK_REGISTRY` | `molfun.modules.blocks` | pairformer, evoformer, simple_transformer |
| `STRUCTURE_MODULE_REGISTRY` | `molfun.modules.structure_module` | ipa, diffusion |
| `EMBEDDER_REGISTRY` | `molfun.modules.embedders` | input, esm |
| `LOSS_REGISTRY` | `molfun.losses` | mse, mae, huber, pearson, openfold |
