# Molfun — Claude Code Context

## Project Overview

**Molfun v0.2.0** is a Python framework for fine-tuning protein structure prediction models (OpenFold/AlphaFold2). It provides a unified API (`MolfunStructureModel`) for training, inference, module swapping, PEFT (LoRA), export, and experiment tracking.

- **Author**: Rubén Cañadas (`rubencr14@gmail.com`)
- **License**: Apache-2.0
- **Python**: ≥ 3.10
- **Core deps**: torch ≥ 2.0, transformers, typer, biopython, numpy

## Architecture

### Logical Layers

**Domain** — Core types, constants, and abstract interfaces:
- `core/` — `TrunkOutput`, `Batch` value objects
- `constants.py` — Amino acid mappings, physicochemical properties, atom definitions
- ABCs: `adapters/base.py`, `training/base.py`, `tracking/base.py`, `losses/base.py`, `modules/*/base.py`
- `modules/registry.py` — Generic `ModuleRegistry` for pluggable components

**Application** — Orchestration and business logic:
- `models/structure.py` — `MolfunStructureModel` facade (predict, fit, swap, export, hub)
- `predict.py` — `predict_structure()`, `predict_properties()`, `predict_affinity()`
- Strategy implementations in `training/` (full, head-only, LoRA, partial fine-tune)
- `pipelines/`, `benchmarks/`, `analysis/`

**Infrastructure** — External integrations:
- `backends/openfold/` — OpenFold adapter
- `data/` — Data loading, RCSB queries, parsers (PDB, mmCIF, SDF, A3M)
- Tracker implementations (wandb, comet, mlflow, langfuse)
- `storage/`, `hub/`, `export/`, `cli/`, `kernels/`, `cache/`

### Key Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Registry** | `ModuleRegistry` | Type-safe plugin system for attention, blocks, embedders, heads |
| **Strategy** | `training/` | Interchangeable fine-tuning strategies (`FinetuneStrategy` ABC) |
| **Adapter** | `adapters/base.py` | Normalize different model backends behind `BaseAdapter` |
| **Template Method** | `FinetuneStrategy.fit()` | Fixed training loop, customizable setup/param_groups |
| **Lazy Import** | `backends/`, `tracking/` | Heavy deps (openfold, wandb) imported only when used |

## How to Add Components

### New Attention Module
1. Subclass `BaseAttention` from `modules/attention/base.py`
2. Decorate with `@ATTENTION_REGISTRY.register("name")`
3. Implement `forward(q, k, v, mask, bias)`, expose `num_heads`, `head_dim`

### New Fine-Tuning Strategy
1. Subclass `FinetuneStrategy` from `training/base.py`
2. Implement `_setup_impl(model)` and `param_groups(model)`
3. The `fit()` template method handles the training loop

### New Model Backend
1. Subclass `BaseAdapter` from `adapters/base.py`
2. Implement `forward()`, `freeze_trunk()`, `unfreeze_trunk()`, `get_trunk_blocks()`, `param_summary()`
3. Register in `ADAPTER_REGISTRY`

### New Loss Function
1. Subclass `LossFunction` from `losses/base.py`
2. Implement `forward(preds, targets, batch) → dict[str, Tensor]`
3. Register with `LossRegistry`

### New Tracker
1. Subclass `BaseTracker` from `tracking/base.py`
2. Implement `start_run()`, `log_metrics()`, `log_config()`, `log_artifact()`, `log_text()`, `end_run()`

## Development Commands

```bash
make install-dev     # Install with dev extras
make install-test    # Install test dependencies
make test            # Full test suite
make test-cov        # Tests with coverage report
make test-modules    # Modular architecture tests only
make test-training   # Training strategy tests only
make test-kernels    # GPU kernel tests only
make lint            # Linter checks
```

### Running Tests

```bash
# Standard run (skip GPU/integration/agents/smoke)
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/ --ignore=tests/kernels --ignore=tests/integration --ignore=tests/agents --ignore=tests/smoke -q

# Specific module
pytest tests/modules/ -v

# Skip slow tests
pytest tests/ -m "not slow"
```

### Test Markers
- `slow` — Long-running tests
- `cuda` / `gpu` — Require CUDA GPU
- `integration` — Integration tests (real data, network)

## Testing Conventions

- **Mock model pattern**: Tests use lightweight `nn.Module` mocks instead of real OpenFold models
- **Fixtures** (`tests/conftest.py`): `device` (cuda/cpu), `dtype` (float16), `sample_sequences`, `single_sequence`
- **Integration fixtures**: `SyntheticAffinityDataset`, `make_loader`, `build_custom_model`
- Tests are organized mirroring the package structure under `tests/`

## Code Style

- **Formatter/Linter**: Ruff
- **Line length**: 100
- **Rules**: E, W, F, I (isort), UP, B (bugbear), SIM
- **Ignored**: E501, B008, B905, SIM108, UP007
- **isort**: known-first-party = `["molfun"]`
- **Target**: Python 3.10+

## Main Exports

```python
from molfun import MolfunStructureModel, OpenFold
from molfun import predict_structure, predict_properties, predict_affinity
```

## CLI

Entry point: `molfun` (via `molfun.cli:app`, built with Typer)

## Architecture Decisions

- **No physical DDD restructure**: The current package layout (adapters/, modules/, training/, data/, etc.) maps logically to DDD layers but is not physically reorganized into domain/application/infrastructure folders. Do not suggest restructuring the directory layout.

## Future Plans

- **ESMFold** backend — ESM-based structure prediction
- **Protenix** backend — Additional structure prediction
- **Docking models** — Protein-ligand docking support

New backends should follow the `BaseAdapter` interface and register in `ADAPTER_REGISTRY`.
