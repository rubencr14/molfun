# Modular Architecture Guide

Molfun's `molfun.modules` package provides a plug-and-play system for protein ML research. Every major component of a structure prediction model — attention, trunk blocks, structure module, input embedder — can be swapped, combined, and extended without touching the training infrastructure.

This guide covers:

1. [Architecture overview](#architecture-overview) — how the pieces fit together
2. [Registries](#registries) — discovering and building components by name
3. [Swapping modules in pre-trained models](#swapping-modules-in-pre-trained-models) — modify OpenFold without losing weights
4. [Building custom models from scratch](#building-custom-models-from-scratch) — compose new architectures
5. [Writing your own module](#writing-your-own-module) — extend the framework
6. [Training custom models](#training-custom-models) — use Molfun's full training stack on your designs
7. [Recipes](#recipes) — concrete research scenarios

---

## Architecture overview

A protein structure prediction model in Molfun is composed of three stages:

```
Input (sequence, MSA)
       │
       ▼
  ┌──────────┐
  │ Embedder  │   aatype + relpos + MSA → initial representations
  └────┬─────┘
       │  single [B, L, D_s]  +  pair [B, L, L, D_p]
       ▼
  ┌──────────┐
  │  Blocks   │   N × (attention + pair ops + transitions)
  │ (×N)      │   Each block refines single + pair representations
  └────┬─────┘
       │  refined single + pair
       ▼
  ┌──────────────────┐
  │ Structure Module  │   representations → 3D coordinates
  └────────┬─────────┘
           │  positions [B, L, 3]
           ▼
      ┌────────┐
      │  Head   │   coordinates/repr → task prediction (affinity, loss, etc.)
      └────────┘
```

Each box is a **pluggable module** with an abstract base class, a registry, and multiple built-in implementations:

| Component | Base class | Registry | Built-in implementations |
|-----------|-----------|----------|-------------------------|
| Attention | `BaseAttention` | `ATTENTION_REGISTRY` | `standard`, `flash`, `linear`, `gated` |
| Block | `BaseBlock` | `BLOCK_REGISTRY` | `evoformer`, `pairformer`, `simple_transformer` |
| Structure Module | `BaseStructureModule` | `STRUCTURE_MODULE_REGISTRY` | `ipa`, `diffusion` |
| Embedder | `BaseEmbedder` | `EMBEDDER_REGISTRY` | `input`, `esm` |

The attention implementations deserve special mention because they are injected *inside* blocks. When you create a block, you pass `attention_cls="flash"` and the block uses that attention mechanism internally. This means you can test a new attention variant across all block types with zero code changes.

---

## Registries

Every module family has a global registry. You can query what's available, build modules by name, and register your own implementations.

### Listing available modules

```python
from molfun.modules import (
    ATTENTION_REGISTRY,
    BLOCK_REGISTRY,
    STRUCTURE_MODULE_REGISTRY,
    EMBEDDER_REGISTRY,
)

print("Attention:  ", ATTENTION_REGISTRY.list())
# ['flash', 'gated', 'linear', 'standard']

print("Blocks:     ", BLOCK_REGISTRY.list())
# ['evoformer', 'pairformer', 'simple_transformer']

print("Structure:  ", STRUCTURE_MODULE_REGISTRY.list())
# ['diffusion', 'ipa']

print("Embedders:  ", EMBEDDER_REGISTRY.list())
# ['esm', 'input']
```

### Building a module by name

```python
from molfun.modules.attention import ATTENTION_REGISTRY

# Build a FlashAttention instance
attn = ATTENTION_REGISTRY.build("flash", num_heads=8, head_dim=32)
print(attn.num_heads)   # 8
print(attn.head_dim)    # 32
print(attn.embed_dim)   # 256

# Build a structure module
from molfun.modules.structure_module import STRUCTURE_MODULE_REGISTRY

sm = STRUCTURE_MODULE_REGISTRY.build(
    "ipa",
    d_single=384,
    d_pair=128,
    n_heads=12,
    n_layers=8,
)
```

### Registering a custom module

```python
from molfun.modules.attention import ATTENTION_REGISTRY, BaseAttention

@ATTENTION_REGISTRY.register("my_sparse_attention")
class SparseAttention(BaseAttention):
    """Only attend to the K nearest neighbors."""

    def __init__(self, num_heads=8, head_dim=32, k=64, **kwargs):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = head_dim
        self.k = k

    def forward(self, q, k, v, mask=None, bias=None):
        # Your sparse attention implementation here
        ...

    @property
    def num_heads(self): return self._num_heads

    @property
    def head_dim(self): return self._head_dim
```

After registration, it's immediately available everywhere:

```python
# Use in any block
block = BLOCK_REGISTRY.build(
    "pairformer",
    d_single=256,
    d_pair=128,
    attention_cls="my_sparse_attention",
)

# Or via ModelBuilder
model = ModelBuilder(
    block="pairformer",
    block_config={"attention_cls": "my_sparse_attention"},
    ...
).build()
```

---

## Swapping modules in pre-trained models

The most common research workflow: start with a pre-trained model (OpenFold), replace one component, and fine-tune. `ModuleSwapper` and `MolfunStructureModel.swap()` make this straightforward.

### Example 1: Replace the structure module in OpenFold

Suppose you've designed a new structure prediction module and want to test it on top of OpenFold's pre-trained Evoformer representations.

```python
from openfold.config import model_config
from molfun.models.structure import MolfunStructureModel
from molfun.modules.structure_module import DiffusionStructureModule

# Load pre-trained OpenFold
model = MolfunStructureModel(
    "openfold",
    config=model_config("model_1_ptm"),
    weights="~/.molfun/weights/finetuning_ptm_2.pt",
    device="cuda",
    head="affinity",
    head_config={"single_dim": 384},
)

# Replace the structure module with a diffusion-based one
old_sm = model.swap(
    "structure_module",
    DiffusionStructureModule(
        d_single=384,
        d_pair=128,
        d_model=256,
        n_layers=4,
        n_steps=50,
    ),
)
# old_sm is the original IPA module (returned for reference)

# The rest of the model (Evoformer, embedder) keeps its pre-trained weights.
# Only the new structure module is randomly initialized.
# Fine-tune:
from molfun.training import PartialFinetune

strategy = PartialFinetune(
    unfreeze_last_n=0,              # keep Evoformer frozen
    unfreeze_structure_module=True,  # train the new module
    lr_trunk=1e-4,
    lr_head=1e-3,
)
history = model.fit(train_loader, val_loader, strategy=strategy, epochs=20)
```

### Example 2: Discover what's inside a model

Before swapping, you might want to inspect the model's internal structure:

```python
# List all modules
modules = model.discover_modules()
for name, mod in modules[:20]:
    print(f"{name:50s} → {type(mod).__name__}")

# Filter by pattern
attention_modules = model.discover_modules(pattern="msa_att|pair_att")
print(f"Found {len(attention_modules)} attention modules")
```

### Example 3: Swap all attention layers at once

Replace every attention module in the Evoformer with FlashAttention for faster training:

```python
from molfun.modules.attention import FlashAttention

n_swapped = model.swap_all(
    pattern=r"msa_att",
    factory=lambda name, old: FlashAttention(
        num_heads=old.num_heads,
        head_dim=old.embed_dim // old.num_heads,
    ),
)
print(f"Swapped {n_swapped} attention modules to FlashAttention")
```

### Example 4: Swap by module type

If you want to replace all instances of a specific PyTorch class:

```python
from molfun.modules.swapper import ModuleSwapper
import torch.nn as nn

# Replace all ReLU activations with SiLU
n = ModuleSwapper.swap_by_type(
    model.adapter.model,
    old_type=nn.ReLU,
    factory=lambda name, old: nn.SiLU(),
)
print(f"Replaced {n} ReLU → SiLU")
```

### Example 5: Transfer weights when swapping

When the replacement module has the same parameter shapes, you can automatically transfer weights from the old module:

```python
model.swap(
    "evoformer.blocks.47",     # replace last Evoformer block
    MyCustomBlock(d_msa=256, d_pair=128),
    transfer_weights=True,     # copy matching parameters from old block
)
```

---

## Building custom models from scratch

For more radical experiments — entirely new architectures — use `ModelBuilder` to compose a model from registered components. The resulting model integrates with Molfun's full training stack (strategies, heads, losses, EMA, checkpointing).

### Example 1: AlphaFold3-style Pairformer

```python
from molfun.modules.builder import ModelBuilder
from molfun.models.structure import MolfunStructureModel

# Build an AF3-style model: single+pair track with Pairformer blocks
built = ModelBuilder(
    embedder="input",
    embedder_config={
        "d_single": 256,
        "d_pair": 128,
        "d_msa": 256,
    },
    block="pairformer",
    block_config={
        "d_single": 256,
        "d_pair": 128,
        "n_heads": 8,
        "n_heads_pair": 4,
        "attention_cls": "flash",   # use FlashAttention inside each block
    },
    n_blocks=24,
    structure_module="ipa",
    structure_module_config={
        "d_single": 256,
        "d_pair": 128,
        "n_heads": 8,
        "n_layers": 8,
    },
).build()

# Wrap in MolfunStructureModel for training
model = MolfunStructureModel.from_custom(
    built,
    device="cuda",
    head="affinity",
    head_config={"single_dim": 256, "hidden_dim": 128},
)

# Train with any strategy
from molfun.training import FullFinetune

strategy = FullFinetune(lr=1e-4, lr_head=1e-3, warmup_steps=500)
history = model.fit(train_loader, val_loader, strategy=strategy, epochs=30)
```

### Example 2: Diffusion structure prediction model

Test a denoising-diffusion approach to coordinate generation:

```python
built = ModelBuilder(
    embedder="input",
    embedder_config={"d_single": 128, "d_pair": 64, "d_msa": 128},
    block="pairformer",
    block_config={
        "d_single": 128,
        "d_pair": 64,
        "n_heads": 4,
        "n_heads_pair": 2,
    },
    n_blocks=8,
    structure_module="diffusion",          # diffusion instead of IPA
    structure_module_config={
        "d_single": 128,
        "d_pair": 64,
        "d_model": 128,
        "n_layers": 4,
        "n_steps": 100,
        "noise_schedule": "cosine",
    },
).build()

model = MolfunStructureModel.from_custom(
    built, device="cuda",
    head="structure",
    head_config={"loss_config": config.loss},
)
```

### Example 3: Lightweight single-track model

For quick experiments or when you don't need a pair track (ESMFold-style):

```python
built = ModelBuilder(
    embedder="input",
    embedder_config={"d_single": 512, "d_pair": 128, "d_msa": 512},
    block="simple_transformer",
    block_config={
        "d_single": 512,
        "n_heads": 8,
        "use_swiglu": True,
        "attention_cls": "flash",
    },
    n_blocks=12,
    structure_module="ipa",
    structure_module_config={"d_single": 512, "d_pair": 128, "n_layers": 4},
).build()

print(built.param_summary())
# {'total': ..., 'trainable': ..., 'frozen': 0}
```

### Example 4: Evoformer with gated attention

Test AlphaFold2's gated self-attention mechanism in a smaller model:

```python
built = ModelBuilder(
    embedder="input",
    embedder_config={"d_single": 128, "d_pair": 64, "d_msa": 128},
    block="evoformer",
    block_config={
        "d_msa": 128,
        "d_pair": 64,
        "n_heads_msa": 4,
        "n_heads_pair": 2,
        "attention_cls": "gated",   # gated attention throughout
    },
    n_blocks=4,
    structure_module="ipa",
    structure_module_config={"d_single": 128, "d_pair": 64, "n_layers": 4},
).build()
```

---

## Writing your own module

The key to the modular system is the abstract base classes. Implement the required methods, register your class, and it becomes available everywhere.

### Custom attention mechanism

Here's a complete example of a custom attention that uses relative position bias:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from molfun.modules.attention import ATTENTION_REGISTRY, BaseAttention


@ATTENTION_REGISTRY.register("relative_position")
class RelativePositionAttention(BaseAttention):
    """Attention with learned relative position bias (ALiBi-inspired)."""

    def __init__(self, num_heads=8, head_dim=32, max_len=2048, **kwargs):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._scale = 1.0 / math.sqrt(head_dim)

        # Learned relative position bias per head
        self.rel_bias = nn.Embedding(2 * max_len + 1, num_heads)
        self.max_len = max_len

    def forward(self, q, k, v, mask=None, bias=None):
        B, H, Lq, D = q.shape
        Lk = k.shape[2]

        logits = torch.matmul(q, k.transpose(-2, -1)) * self._scale

        # Add relative position bias
        positions = torch.arange(Lq, device=q.device)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel_pos = rel_pos.clamp(-self.max_len, self.max_len) + self.max_len
        pos_bias = self.rel_bias(rel_pos).permute(2, 0, 1).unsqueeze(0)  # [1, H, L, L]
        logits = logits + pos_bias

        if bias is not None:
            logits = logits + bias
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))

        return torch.matmul(F.softmax(logits, dim=-1), v)

    @property
    def num_heads(self): return self._num_heads

    @property
    def head_dim(self): return self._head_dim


# Now use it:
model = ModelBuilder(
    block="pairformer",
    block_config={"attention_cls": "relative_position", "d_single": 256, "d_pair": 128},
    ...
).build()
```

### Custom structure module

Implement `BaseStructureModule` to create a new coordinate prediction approach:

```python
import torch
import torch.nn as nn
from molfun.modules.structure_module import (
    STRUCTURE_MODULE_REGISTRY,
    BaseStructureModule,
    StructureModuleOutput,
)


@STRUCTURE_MODULE_REGISTRY.register("equivariant")
class EquivariantStructureModule(BaseStructureModule):
    """SE(3)-equivariant structure prediction (research scaffold)."""

    def __init__(self, d_single=384, d_pair=128, n_layers=4, **kwargs):
        super().__init__()
        self._d_single = d_single
        self._d_pair = d_pair

        self.layers = nn.ModuleList([
            EquivariantLayer(d_single, d_pair)
            for _ in range(n_layers)
        ])
        self.coord_proj = nn.Linear(d_single, 3)

    def forward(self, single, pair, aatype=None, mask=None):
        for layer in self.layers:
            single = layer(single, pair, mask)

        positions = self.coord_proj(single)
        return StructureModuleOutput(
            positions=positions,
            single_repr=single,
            confidence=torch.sigmoid(single.mean(dim=-1)),
        )

    @property
    def d_single(self): return self._d_single

    @property
    def d_pair(self): return self._d_pair
```

### Custom block

Implement `BaseBlock` to create a new trunk architecture:

```python
from molfun.modules.blocks import BLOCK_REGISTRY, BaseBlock
from molfun.modules.blocks.base import BlockOutput


@BLOCK_REGISTRY.register("geometric_transformer")
class GeometricTransformerBlock(BaseBlock):
    """Transformer block with geometric features (distance, angles)."""

    def __init__(self, d_single=256, d_pair=128, n_heads=8, **kwargs):
        super().__init__()
        self._d_single = d_single
        self._d_pair = d_pair
        # ... your layers here ...

    def forward(self, single, pair=None, mask=None, pair_mask=None):
        # ... your forward pass ...
        return BlockOutput(single=updated_single, pair=updated_pair)

    @property
    def d_single(self): return self._d_single

    @property
    def d_pair(self): return self._d_pair
```

---

## Training custom models

Models built with `ModelBuilder` or `from_custom` have full access to Molfun's training infrastructure. Every strategy (HeadOnly, LoRA, Partial, Full) works out of the box.

### LoRA on a custom model

```python
from molfun.modules.builder import ModelBuilder
from molfun.models.structure import MolfunStructureModel
from molfun.training import LoRAFinetune

# Build model
built = ModelBuilder(
    embedder="input",
    embedder_config={"d_single": 256, "d_pair": 128, "d_msa": 256},
    block="pairformer",
    block_config={"d_single": 256, "d_pair": 128, "n_heads": 8, "n_heads_pair": 4},
    n_blocks=12,
    structure_module="ipa",
    structure_module_config={"d_single": 256, "d_pair": 128, "n_layers": 4},
).build()

model = MolfunStructureModel.from_custom(
    built, device="cuda",
    head="affinity",
    head_config={"single_dim": 256, "hidden_dim": 128},
)

# LoRA targets linear layers inside the blocks
strategy = LoRAFinetune(
    rank=8,
    target_modules=["q_proj", "v_proj"],   # match your block's layer names
    lr_lora=1e-4,
    lr_head=1e-3,
    warmup_steps=100,
    ema_decay=0.999,
)
history = model.fit(train_loader, val_loader, strategy=strategy, epochs=20)

# Save and export
model.save("checkpoints/custom_pairformer_lora/")
model.merge()  # fuse LoRA weights for deployment
```

### Comparing architectures

Run a controlled experiment comparing different trunk architectures:

```python
import torch
from molfun.modules.builder import ModelBuilder
from molfun.models.structure import MolfunStructureModel
from molfun.training import HeadOnlyFinetune

CONFIGS = {
    "evoformer_gated": {
        "block": "evoformer",
        "block_config": {"d_msa": 128, "d_pair": 64, "n_heads_msa": 4,
                         "n_heads_pair": 2, "attention_cls": "gated"},
    },
    "pairformer_flash": {
        "block": "pairformer",
        "block_config": {"d_single": 128, "d_pair": 64, "n_heads": 4,
                         "n_heads_pair": 2, "attention_cls": "flash"},
    },
    "simple_linear": {
        "block": "simple_transformer",
        "block_config": {"d_single": 128, "n_heads": 4,
                         "attention_cls": "linear"},
    },
}

results = {}
for name, cfg in CONFIGS.items():
    torch.manual_seed(42)

    built = ModelBuilder(
        embedder="input",
        embedder_config={"d_single": 128, "d_pair": 64, "d_msa": 128},
        n_blocks=4,
        structure_module="ipa",
        structure_module_config={"d_single": 128, "d_pair": 64, "n_layers": 2},
        **cfg,
    ).build()

    model = MolfunStructureModel.from_custom(
        built, device="cuda",
        head="affinity",
        head_config={"single_dim": 128},
    )

    strategy = HeadOnlyFinetune(lr=1e-3, ema_decay=0.999)
    history = model.fit(train_loader, val_loader, strategy=strategy, epochs=10)

    results[name] = {
        "final_val_loss": history[-1].get("val_loss"),
        "params": built.param_summary()["total"],
    }
    print(f"{name}: val_loss={results[name]['final_val_loss']:.4f}, "
          f"params={results[name]['params']:,}")
```

---

## Recipes

### Recipe 1: "I want to test if FlashAttention speeds up my OpenFold fine-tuning"

```python
model = MolfunStructureModel("openfold", config=cfg, weights=weights_path,
                              head="affinity", head_config={"single_dim": 384})

# Swap all attention in the Evoformer
from molfun.modules.attention import FlashAttention
model.swap_all(
    pattern=r"mha",
    factory=lambda name, old: FlashAttention(num_heads=8, head_dim=32),
)

# Fine-tune as usual
strategy = LoRAFinetune(rank=8, lr_lora=1e-4)
history = model.fit(train_loader, val_loader, strategy=strategy, epochs=20)
```

### Recipe 2: "I want to compare IPA vs diffusion for structure prediction"

```python
for sm_name in ["ipa", "diffusion"]:
    built = ModelBuilder(
        embedder="input",
        embedder_config={"d_single": 128, "d_pair": 64, "d_msa": 128},
        block="pairformer",
        block_config={"d_single": 128, "d_pair": 64, "n_heads": 4, "n_heads_pair": 2},
        n_blocks=4,
        structure_module=sm_name,
        structure_module_config={"d_single": 128, "d_pair": 64, "n_layers": 4},
    ).build()

    model = MolfunStructureModel.from_custom(built, head="affinity",
                                              head_config={"single_dim": 128})
    # train and compare...
```

### Recipe 3: "I want to prototype a new attention mechanism quickly"

```python
from molfun.modules.attention import ATTENTION_REGISTRY, BaseAttention

@ATTENTION_REGISTRY.register("my_experiment")
class MyExperimentAttention(BaseAttention):
    def __init__(self, num_heads=8, head_dim=32, **kwargs):
        super().__init__()
        self._num_heads, self._head_dim = num_heads, head_dim
        # ... your experimental layers ...

    def forward(self, q, k, v, mask=None, bias=None):
        # ... your experimental logic ...
        return output

    @property
    def num_heads(self): return self._num_heads
    @property
    def head_dim(self): return self._head_dim

# Test it immediately in a full model
built = ModelBuilder(
    block="pairformer",
    block_config={"d_single": 64, "d_pair": 32, "n_heads": 4,
                  "n_heads_pair": 2, "attention_cls": "my_experiment"},
    embedder="input",
    embedder_config={"d_single": 64, "d_pair": 32, "d_msa": 64},
    n_blocks=2,
    structure_module="ipa",
    structure_module_config={"d_single": 64, "d_pair": 32, "n_layers": 2},
).build()

# Quick sanity check
batch = {"aatype": torch.randint(0, 20, (2, 32)),
         "residue_index": torch.arange(32).unsqueeze(0).expand(2, -1)}
output = built(batch)
print(output.single_repr.shape)       # [2, 32, 64]
print(output.structure_coords.shape)  # [2, 32, 3]
```

### Recipe 4: "I want to build an ESMFold-like model from ESM-2 embeddings"

```python
# Requires: pip install fair-esm

built = ModelBuilder(
    embedder="esm",
    embedder_config={
        "esm_model": "esm2_t33_650M_UR50D",
        "d_single": 384,
        "d_pair": 128,
        "freeze_lm": True,      # keep ESM-2 frozen
    },
    block="simple_transformer",
    block_config={
        "d_single": 384,
        "n_heads": 12,
        "use_swiglu": True,
        "attention_cls": "flash",
    },
    n_blocks=8,
    structure_module="ipa",
    structure_module_config={
        "d_single": 384,
        "d_pair": 128,
        "n_layers": 8,
    },
).build()

model = MolfunStructureModel.from_custom(
    built, device="cuda",
    head="affinity",
    head_config={"single_dim": 384},
)
```

---

## Summary

| What you want to do | Tool to use | Effort |
|---------------------|-------------|--------|
| Speed up attention in OpenFold | `model.swap_all(pattern, factory)` | 3 lines |
| Replace the structure module | `model.swap("structure_module", new)` | 1 line |
| Build a Pairformer model from scratch | `ModelBuilder(block="pairformer", ...)` | 10 lines |
| Test a new attention mechanism | `@ATTENTION_REGISTRY.register(...)` | ~30 lines |
| Compare architectures | Loop over `ModelBuilder` configs | 20 lines |
| Use ESM-2 as the backbone | `ModelBuilder(embedder="esm", ...)` | 10 lines |
| Train any custom model | `MolfunStructureModel.from_custom(built)` | Same as OpenFold |

The modular system is designed to make the gap between "I have an idea" and "I have a running experiment" as small as possible. Build your component, register it, plug it in, and train with production-grade infrastructure.
