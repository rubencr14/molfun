# Molfun — Fine-Tuning & Modular Architecture for Molecular ML

![Molfun Banner](./docs/banner.png)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#installation)
[![CUDA](https://img.shields.io/badge/CUDA-12%2B-green.svg)](#requirements)

**Molfun** is an open-source framework for **fine-tuning protein ML models** with a **modular, plug-and-play architecture** for research and experimentation. It provides a unified interface to adapt pre-trained structure prediction models (OpenFold/AlphaFold2, and in the future ESMFold, Protenix, docking models, etc.) to specific tasks like binding affinity prediction, while making it easy to swap, combine, and extend every internal component — attention mechanisms, trunk blocks, structure modules, and embedders.

---

## What problem does Molfun solve?

Pre-trained protein models like AlphaFold2 contain enormous amounts of structural knowledge, but using them for **specific scientific tasks** (predicting binding affinity, classifying protein families, scoring docking poses) requires fine-tuning — and doing it well is surprisingly hard.

You need to decide **what to freeze**, **how to inject trainable parameters**, **which learning rate schedule** to use, and how to avoid catastrophic forgetting on a small dataset. You also need proper EMA, gradient accumulation, and early stopping to get stable results. And if you want to experiment with new architectural ideas — a different attention mechanism, a diffusion-based structure module — you shouldn't have to rewrite the training loop.

Molfun handles all of this:

```python
from molfun.models.structure import MolfunStructureModel
from molfun.training import LoRAFinetune

model = MolfunStructureModel("openfold", config=cfg, weights="weights.pt",
                             head="affinity", head_config={"single_dim": 384})

strategy = LoRAFinetune(rank=8, lr_lora=1e-4, lr_head=1e-3, ema_decay=0.999)
history = model.fit(train_loader, val_loader, strategy=strategy, epochs=30)
```

Or build entirely custom architectures from modular components:

```python
from molfun.modules.builder import ModelBuilder

model = ModelBuilder(
    embedder="input", block="pairformer", structure_module="ipa",
    n_blocks=24,
    block_config={"d_single": 256, "d_pair": 128, "attention_cls": "flash"},
).build()
```

---

## Core Capabilities

### 1. Model Fine-Tuning

The fine-tuning framework is the heart of Molfun. It is designed around **strategy classes** that encapsulate all the complexity of adapting a large pre-trained model to a downstream task.

**Four strategies** cover the full spectrum from conservative to aggressive:

| Strategy | Trainable Params | Best For | Key Idea |
|----------|-----------------|----------|----------|
| **HeadOnly** | ~50K (head only) | < 100 samples | Freeze everything, train a prediction head on top of frozen representations |
| **LoRA** | ~600K (adapters + head) | 100 – 5K samples | Inject low-rank matrices into attention layers. 99.3% of the model stays frozen. |
| **Partial** | ~5M (last N blocks + head) | 1K – 10K samples | Unfreeze the last N Evoformer blocks so the model can adapt its task-specific layers |
| **Full** | ~93M (everything) | > 10K samples | Unfreeze all parameters with layer-wise LR decay to keep early layers stable |

Every strategy includes **warmup scheduling**, **cosine/linear LR decay**, **EMA** (exponential moving average for stable inference on small datasets), **gradient accumulation** (to simulate large batches on a single GPU), **gradient clipping**, **mixed precision**, and **early stopping**.

The model wrapper (`MolfunStructureModel`) is backend-agnostic. Today it supports OpenFold; adding ESMFold, Protenix, or any new model means implementing a single adapter class.

### 2. Modular Architecture

Molfun's `molfun.modules` package provides a **plug-and-play system** for protein ML research. Every major component has an abstract base class, a registry, and multiple built-in implementations:

| Component | Registry | Built-in implementations |
|-----------|----------|-------------------------|
| **Attention** | `ATTENTION_REGISTRY` | `standard`, `flash`, `linear`, `gated` |
| **Block** | `BLOCK_REGISTRY` | `evoformer`, `pairformer`, `simple_transformer` |
| **Structure Module** | `STRUCTURE_MODULE_REGISTRY` | `ipa`, `diffusion` |
| **Embedder** | `EMBEDDER_REGISTRY` | `input` (AF2-style), `esm` (ESM-2 LM) |

Three workflows are supported:

- **Registry lookup** — build components by name and wire them together
- **Module swapping** — patch components inside pre-trained models at runtime without losing weights
- **Model building** — compose custom architectures declaratively with `ModelBuilder`

```python
# Swap attention in a pre-trained OpenFold
from molfun.modules.attention import FlashAttention
model.swap_all(
    pattern=r"msa_att",
    factory=lambda name, old: FlashAttention(num_heads=8, head_dim=32),
)

# Or build a custom architecture from scratch
from molfun.modules.builder import ModelBuilder
built = ModelBuilder(
    embedder="input",
    block="pairformer",
    block_config={"d_single": 256, "d_pair": 128, "attention_cls": "flash"},
    n_blocks=24,
    structure_module="ipa",
).build()
model = MolfunStructureModel.from_custom(built, head="affinity",
                                          head_config={"single_dim": 256})
```

See [docs/modules.md](docs/modules.md) for the full modular architecture guide with tutorials and examples.

### 3. GPU Kernels

Molfun provides Triton GPU kernels for geometric primitives that molecular ML pipelines depend on:

| Kernel | Speedup vs CPU | Notes |
|--------|---------------|-------|
| **Batch RMSD** (Kabsch) | 800x vs MDAnalysis, 20x vs MDTraj | Full superposition alignment |
| **Contact Maps** (bit-packed) | 45x vs MDAnalysis, 36x vs MDTraj | 8x less memory than boolean matrices |
| **Pairwise Distances** | GPU-native | Foundation for contact maps and graph construction |

Additional Triton kernels for model internals: fused LayerNorm, GELU, fused linear+bias+residual.

### 4. Pluggable Loss Registry

Losses are first-class objects registered by name and fully decoupled from heads and training strategies.

```python
from molfun.losses import LOSS_REGISTRY

loss_fn = LOSS_REGISTRY["huber"]()
result  = loss_fn(preds, targets)

# Register a custom loss:
from molfun.losses import LossFunction

@LOSS_REGISTRY.register("tmscore")
class TMScoreLoss(LossFunction):
    def forward(self, preds, targets=None, batch=None): ...
```

Built-in losses: `mse`, `mae`, `huber`, `pearson` (affinity) · `openfold` (FAPE + auxiliary structure losses).

### 5. Data Pipeline

A complete data pipeline for protein fine-tuning tasks:

- **PDBFetcher** — download structures from RCSB
- **AffinityFetcher** — parse PDBbind index files or CSV datasets
- **MSAProvider** — generate or load pre-computed MSAs (single-sequence, A3M, or MMseqs2)
- **OpenFoldFeaturizer** — convert PDB/mmCIF files to full OpenFold feature dicts
- **StructureDataset / AffinityDataset** — PyTorch datasets with on-the-fly or pre-computed features
- **DataSplitter** — random, temporal, sequence-identity, or family-based splits

### 6. MD Analysis

The `molfun.analysis` module provides a GPU-accelerated interface for molecular dynamics trajectory analysis. It loads trajectories (DCD, XTC, etc.), transfers coordinates to GPU, and runs all analysis with Triton kernels.

---

## Supported Models

| Model | Status | Adapter |
|-------|--------|---------|
| **OpenFold** (AlphaFold2) | Fully supported | `OpenFoldAdapter` |
| ESMFold | Planned | — |
| Protenix | Planned | — |
| DiffDock / ML Docking | Planned | — |

Adding a new model requires implementing `BaseAdapter` (forward, freeze/unfreeze, PEFT targets) — typically ~100 lines.

---

## Architecture

```
molfun/
├── models/          # MolfunStructureModel — unified model wrapper
├── adapters/        # Backend adapters (OpenFold, future: ESMFold, ...)
├── modules/         # Pluggable components for research
│   ├── attention/   #   Standard, Flash, Linear, Gated attention
│   ├── blocks/      #   Evoformer, Pairformer, SimpleTransformer
│   ├── structure_module/  # IPA, Diffusion
│   ├── embedders/   #   Input (AF2-style), ESM (ESM-2 LM)
│   ├── registry.py  #   Generic module registry
│   ├── swapper.py   #   Runtime module replacement
│   └── builder.py   #   Declarative model composition
├── training/        # Fine-tuning strategies (HeadOnly, LoRA, Partial, Full)
├── peft/            # LoRA / IA3 injection (builtin + HuggingFace PEFT)
├── heads/           # Task heads (AffinityHead, StructureLossHead, ...)
├── losses/          # Loss registry: mse, mae, huber, pearson, openfold
├── helpers/         # Shared utilities: EMA, scheduler, batch helpers
├── featurizers/     # OpenFoldFeaturizer — PDB/mmCIF → feature dict
├── data/            # Datasets, data sources, splits, MSA handling
├── kernels/         # Triton GPU kernels (RMSD, contact maps, distances, ...)
├── analysis/        # MD trajectory analysis (GPU-accelerated)
└── cli/             # CLI entry point (molfun structure / affinity / info)

docs/                # Guides and tutorials
tests/               # Unit + GPU integration tests
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/rubencr14/molfun.git
cd molfun

# Install Molfun (registers the `molfun` CLI command):
pip install -e .

# Install OpenFold (requires CUDA GPU):
pip install torch --index-url https://download.pytorch.org/whl/cu124
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 pip install git+https://github.com/aqlaboratory/openfold.git
pip install dm-tree ml-collections biopython

# Download pre-trained weights (~370 MB):
mkdir -p ~/.molfun/weights
wget -O ~/.molfun/weights/finetuning_ptm_2.pt \
    https://huggingface.co/nz/openfold/resolve/main/finetuning_ptm_2.pt
```

### Verify Installation

```bash
molfun info
```

```
Python    : 3.10.x
PyTorch   : 2.x | CUDA 12.x
GPU       : NVIDIA RTX 5090 | 31 GB VRAM

Weights   : ~/.molfun/weights/finetuning_ptm_2.pt  [✓]
OpenFold  : installed

Losses    : huber, mae, mse, openfold, pearson
Heads     : affinity, structure
Strategies: lora, partial, full, head_only
Modules   : 4 attention, 3 blocks, 2 structure_modules, 2 embedders
```

### Fine-Tune on Protein Structures (no labels needed)

Fine-tune OpenFold directly on PDB coordinates using FAPE loss — no affinity labels required:

```bash
molfun structure data/pdbs/ --strategy lora --epochs 20 --lr 2e-4
```

Or programmatically:

```python
from molfun.models.structure import MolfunStructureModel
from molfun.training import LoRAFinetune

model = MolfunStructureModel(
    "openfold", config=cfg, weights="weights.pt",
    head="structure", head_config={"loss_config": cfg.loss},
)
strategy = LoRAFinetune(rank=8, lr_lora=2e-4, ema_decay=0.999)
history = model.fit(train_loader, val_loader, strategy=strategy, epochs=20)
```

### Fine-Tune on Binding Affinity

```bash
molfun affinity data/pdbs/ --data data/pdbbind.csv --strategy lora --epochs 30
```

Or programmatically:

```python
model = MolfunStructureModel(
    "openfold", config=cfg, weights="weights.pt",
    head="affinity", head_config={"single_dim": 384},
)
strategy = LoRAFinetune(rank=8, lr_lora=1e-4, lr_head=1e-3, ema_decay=0.999)
history = model.fit(train_loader, val_loader, strategy=strategy, epochs=30)
```

### Build a Custom Architecture

```python
from molfun.modules.builder import ModelBuilder
from molfun.models.structure import MolfunStructureModel

built = ModelBuilder(
    embedder="input",
    embedder_config={"d_single": 256, "d_pair": 128, "d_msa": 256},
    block="pairformer",
    block_config={"d_single": 256, "d_pair": 128, "attention_cls": "flash"},
    n_blocks=12,
    structure_module="ipa",
    structure_module_config={"d_single": 256, "d_pair": 128},
).build()

model = MolfunStructureModel.from_custom(
    built, head="affinity", head_config={"single_dim": 256},
)
strategy = LoRAFinetune(rank=8, lr_lora=1e-4, lr_head=1e-3)
history = model.fit(train_loader, val_loader, strategy=strategy, epochs=20)
```

See [docs/openfold/run.md](docs/openfold/run.md) for the full production fine-tuning guide and [docs/modules.md](docs/modules.md) for the modular architecture tutorial.

---

## Documentation

| Guide | Description |
|-------|-------------|
| [docs/openfold/run.md](docs/openfold/run.md) | Production fine-tuning guide: strategies, hyperparameters, checkpointing |
| [docs/modules.md](docs/modules.md) | Modular architecture tutorial: registries, swapping, building, custom modules |

---

## Performance

### GPU Kernels

Tested on NVIDIA RTX 5090, CUDA 12.8, PyTorch 2.x.

**RMSD with Kabsch alignment** (2501 frames, 3891 atoms):

| Method | Time | vs Molfun |
|--------|------|-----------|
| **Molfun (Triton)** | **0.71 ms** | — |
| PyTorch GPU | 2.21 ms | 3.1x slower |
| MDTraj (C/Cython) | 14.69 ms | 21x slower |
| MDAnalysis | 565.98 ms | 800x slower |

**Contact Maps** (2501 frames, 254 Cα atoms, 8.0 Å cutoff):

| Method | Time | vs Molfun |
|--------|------|-----------|
| **Molfun (Triton)** | **66 ms** | — |
| PyTorch GPU | 71 ms | 1.1x slower |
| MDTraj | 2,432 ms | 37x slower |
| MDAnalysis | 3,163 ms | 48x slower |

### Fine-Tuning

All four strategies verified with gradient flow tests on real OpenFold (93M params) with pre-trained weights. LoRA fine-tuning of OpenFold takes ~5 seconds per epoch on synthetic data (seq_len=32) on a single GPU.

---

## Tests

```bash
# Full test suite
python -m pytest tests/ -v

# Fine-tuning tests (mock + real GPU)
python -m pytest tests/training/ tests/models/ -v

# Module tests (attention, blocks, builder, swapper)
python -m pytest tests/modules/ -v
```

## License

Apache 2.0 — see [LICENSE](./LICENSE).

## Citation

```bibtex
@software{molfun,
  title  = {Molfun: Fine-Tuning \& Modular Architecture for Molecular ML},
  author = {Rubén Cañadas},
  year   = {2026},
  url    = {https://github.com/rubencr14/molfun/}
}
```
