# Molfun — Fine-Tuning & GPU Acceleration for Molecular ML

![Molfun Banner](./docs/banner.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#installation)
[![CUDA](https://img.shields.io/badge/CUDA-12%2B-green.svg)](#requirements)

**Molfun** is an open-source framework for **fine-tuning protein ML models** and **accelerating molecular simulations** on GPU. It provides a unified interface to adapt pre-trained structure prediction models (OpenFold/AlphaFold2, and in the future ESMFold, Protenix, docking models, etc.) to specific tasks like binding affinity prediction, with production-grade training infrastructure and high-performance Triton kernels.

---

## What problem does Molfun solve?

Pre-trained protein models like AlphaFold2 contain enormous amounts of structural knowledge, but using them for **specific scientific tasks** (predicting binding affinity, classifying protein families, scoring docking poses) requires fine-tuning — and doing it well is surprisingly hard.

You need to decide **what to freeze**, **how to inject trainable parameters**, **which learning rate schedule** to use, and how to avoid catastrophic forgetting on a small dataset. You also need proper EMA, gradient accumulation, and early stopping to get stable results.

Molfun handles all of this. You pick a strategy, point it at your data, and train:

```python
from molfun.models.structure import MolfunStructureModel
from molfun.training import LoRAFinetune

model = MolfunStructureModel("openfold", config=cfg, weights="weights.pt",
                             head="affinity", head_config={"single_dim": 384})

strategy = LoRAFinetune(rank=8, lr_lora=1e-4, lr_head=1e-3, ema_decay=0.999)
history = model.fit(train_loader, val_loader, strategy=strategy, epochs=30)
```

Beyond fine-tuning, Molfun also provides **GPU-accelerated kernels** for molecular analysis (RMSD, contact maps, distances) that are 10-800x faster than CPU tools, and an **MD analysis module** for trajectory processing.

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

### 2. GPU Kernels

Molfun provides Triton GPU kernels for the geometric primitives that molecular ML pipelines depend on. These aren't just faster — they enable workflows that would be impractical on CPU.

| Kernel | Speedup vs CPU | Notes |
|--------|---------------|-------|
| **Batch RMSD** (Kabsch) | 800x vs MDAnalysis, 20x vs MDTraj | Full superposition alignment |
| **Contact Maps** (bit-packed) | 45x vs MDAnalysis, 36x vs MDTraj | 8x less memory than boolean matrices |
| **Pairwise Distances** | GPU-native | Foundation for contact maps and graph construction |

These kernels are used both in analysis workflows and as building blocks for ML feature computation (contact maps for attention masks, distance features for geometric constraints, etc.).

### 3. MD Analysis

The `molfun.analysis` module provides a GPU-accelerated interface for molecular dynamics trajectory analysis. It loads trajectories (DCD, XTC, etc.), transfers coordinates to GPU, and runs all analysis with Triton kernels — avoiding the CPU bottleneck that makes tools like MDAnalysis slow on large trajectories.

### 4. Data Pipeline

A complete data pipeline for protein fine-tuning tasks:

- **PDBFetcher** — download structures from RCSB
- **AffinityFetcher** — parse PDBbind index files or CSV datasets
- **MSAProvider** — generate or load pre-computed MSAs (single-sequence, A3M, or MMseqs2)
- **StructureDataset / AffinityDataset** — PyTorch datasets with on-the-fly or pre-computed features
- **DataSplitter** — random, temporal, sequence-identity, or family-based splits (the latter two prevent data leakage from homologous sequences)

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
├── training/        # Fine-tuning strategies (HeadOnly, LoRA, Partial, Full)
├── peft/            # LoRA / IA3 injection (builtin + HuggingFace PEFT)
├── heads/           # Task heads (AffinityHead, future: ClassificationHead, ...)
├── data/            # Datasets, data sources, splits, MSA handling
├── kernels/         # Triton GPU kernels (RMSD, contact maps, distances, ...)
├── analysis/        # MD trajectory analysis (GPU-accelerated)
└── api/             # FastAPI backend (dashboard integration)

scripts/             # CLI tools (fine_tune.py)
docs/                # Guides (docs/openfold/run.md)
tests/               # Unit + GPU integration tests
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/rubencr14/molfun.git
cd molfun

# For fine-tuning OpenFold (requires CUDA GPU):
pip install torch --index-url https://download.pytorch.org/whl/cu124
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 pip install git+https://github.com/aqlaboratory/openfold.git
pip install dm-tree ml-collections biopython

# Download pre-trained weights (~370 MB):
mkdir -p ~/.molfun/weights
wget -O ~/.molfun/weights/finetuning_ptm_2.pt \
    https://huggingface.co/nz/openfold/resolve/main/finetuning_ptm_2.pt
```

### Demo Run

Verify everything works with synthetic data (no external datasets needed):

```bash
PYTHONPATH=. python scripts/fine_tune.py --demo --strategy lora --epochs 3 --output runs/demo/
```

### Fine-Tune on Real Data

```bash
PYTHONPATH=. python scripts/fine_tune.py \
    --data data/pdbbind.csv \
    --pdbs data/pdbs/ \
    --strategy lora \
    --lora-rank 8 \
    --epochs 30 \
    --ema 0.999 \
    --output runs/lora_pdbbind/
```

See [docs/openfold/run.md](docs/openfold/run.md) for the full production fine-tuning guide with hyperparameter recommendations, strategy selection, and deployment instructions.

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
# Full test suite (111 tests: data pipeline, kernels, models, training strategies)
python -m pytest tests/ -v

# Just the fine-tuning tests (mock + real GPU)
python -m pytest tests/training/ tests/models/ -v
```

---

## Roadmap

**Fine-Tuning**
- [ ] ESMFold adapter
- [ ] Classification head (protein family, function prediction)
- [ ] Multi-task heads (affinity + contact prediction jointly)
- [ ] Distributed training (DDP / FSDP)

**Models**
- [ ] Protenix adapter
- [ ] ML docking model adapters (DiffDock, etc.)

**Kernels**
- [ ] Fused SwiGLU / LayerNorm for transformer layers
- [ ] Sparse neighbor / edge list generation
- [ ] Residue-level contact maps for large systems

**Data**
- [ ] Automated PDBbind download + preprocessing
- [ ] OpenFold feature pre-computation pipeline
- [ ] MSA generation via ColabFold/MMseqs2 integration

---

## License

MIT — see [LICENSE](./LICENSE).

## Citation

```bibtex
@software{molfun,
  title  = {Molfun: Fine-Tuning & GPU Acceleration for Molecular ML},
  author = {Rubén Cañadas},
  year   = {2026},
  url    = {https://github.com/rubencr14/molfun/}
}
```
