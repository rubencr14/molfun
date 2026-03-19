---
title: Installation
---

# Installation

This guide covers installing Molfun, its optional extras, and verifying that everything
works correctly.

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | >= 3.10 | Use `python --version` to check |
| **pip** | >= 22.0 | Bundled with modern Python |
| **OS** | Linux, macOS | Windows via WSL2 |

!!! info "GPU support"

    A CUDA-capable GPU is **not required** for inference or small-scale fine-tuning, but it
    is strongly recommended for training. Molfun supports CUDA 11.8+ with PyTorch >= 2.0.

---

## Install Molfun

=== "Basic (CPU)"

    The default install pulls in the core dependencies --- PyTorch (CPU), Transformers,
    Biopython, Typer, and NumPy:

    ```bash
    pip install molfun
    ```

=== "GPU (CUDA)"

    For GPU-accelerated training and inference, install with the `gpu` extra.
    This pulls in the CUDA-enabled PyTorch build and Triton:

    ```bash
    pip install molfun[gpu]
    ```

=== "From Source"

    Clone the repository and install in editable mode:

    ```bash
    git clone https://github.com/rubencr14/molfun.git
    cd molfun
    pip install -e ".[dev]"
    ```

---

## Optional Extras

Molfun ships several extras for different use cases. Combine them as needed --- for example,
`pip install molfun[gpu,dev]`.

| Extra | What it adds | When you need it |
|-------|-------------|-----------------|
| `gpu` | CUDA PyTorch, Triton kernels | Training on GPU, fast RMSD/contact maps |
| `dev` | Ruff, pre-commit, mypy | Contributing to Molfun |
| `test` | pytest, pytest-cov, hypothesis | Running the test suite |
| `docs` | MkDocs Material, mkdocstrings | Building the documentation |
| `tracking` | wandb, comet-ml, mlflow, langfuse | Experiment tracking integrations |
| `all` | Everything above | Full development environment |

---

## Verify the Installation

### Python API

Open a Python shell and confirm the import succeeds:

```bash
python -c "from molfun import MolfunStructureModel; print('OK')"
```

You should see:

```
OK
```

### CLI

The `molfun` command should be available on your `PATH`:

```bash
molfun info
```

This prints the installed version, detected backends, available devices, and registered
modules. Sample output:

```
Molfun v0.2.0
Backends:   openfold
Device:     cpu (CUDA not available)
Strategies: full, head_only, lora, partial
Trackers:   console, wandb, comet, mlflow, langfuse
```

!!! tip "Check GPU detection"

    If you installed with `gpu` extras, `molfun info` should report a CUDA device.
    If it still shows `cpu`, see the troubleshooting section below.

---

## Troubleshooting

### CUDA not detected after installing `molfun[gpu]`

The `gpu` extra installs the CUDA-enabled PyTorch wheel, but your system still needs a
compatible NVIDIA driver.

```bash
# Check driver version
nvidia-smi

# Check PyTorch sees CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` returns `False`:

1. Verify your NVIDIA driver supports CUDA 11.8+ (`nvidia-smi` shows the max CUDA version).
2. Reinstall PyTorch for your specific CUDA version following
   [pytorch.org/get-started](https://pytorch.org/get-started/locally/).
3. Reinstall Molfun after fixing PyTorch: `pip install molfun[gpu]`.

### OpenFold dependency errors

OpenFold has specific version constraints on some packages. If you see build errors:

```bash
# Install OpenFold dependencies first
pip install deepspeed dm-tree ml-collections

# Then install Molfun
pip install molfun
```

!!! warning "Conflicting NumPy versions"

    OpenFold requires NumPy < 2.0. If another package pulls in NumPy 2.x, pin it:

    ```bash
    pip install "numpy>=1.24,<2.0"
    pip install molfun
    ```

### `KMP_DUPLICATE_LIB_OK` error on macOS

If you see `OMP: Error #15: Initializing libiomp5.dylib`, set the environment variable
before running Molfun:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

Or add it to your shell profile (`~/.zshrc` or `~/.bashrc`).

### Import errors for optional trackers

Tracker integrations use lazy imports. If you get an `ImportError` when using
`tracker="wandb"`, install the corresponding package:

```bash
pip install wandb      # for WandB
pip install comet-ml   # for Comet
pip install mlflow     # for MLflow
pip install langfuse   # for Langfuse
```

Or install all trackers at once:

```bash
pip install molfun[tracking]
```

---

## Next Steps

Your installation is ready. Head to the **[Quick Start](quickstart.md)** to run your first
prediction in under five minutes.
