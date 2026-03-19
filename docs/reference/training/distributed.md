# Distributed Training

Molfun supports distributed training via PyTorch's **DistributedDataParallel
(DDP)** and **Fully Sharded Data Parallel (FSDP)** through the
`molfun.training.distributed` module.

## Quick Start

```python
from molfun import MolfunStructureModel

model = MolfunStructureModel.from_pretrained("openfold_v2")

# DDP training (data parallelism)
model.fit(
    train_dataset=ds,
    strategy="full",
    distributed="ddp",
    epochs=10,
)

# FSDP training (model + data parallelism)
model.fit(
    train_dataset=ds,
    strategy="full",
    distributed="fsdp",
    epochs=10,
)
```

## Module Reference

::: molfun.training.distributed
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## DDP -- DistributedDataParallel

Standard data-parallel training. Each GPU holds a full model replica and
processes a shard of the data.

### Launch

```bash
# Using torchrun
torchrun --nproc_per_node=4 train.py

# Using the Molfun CLI
molfun run train.py --gpus 4
```

### Programmatic Usage

```python
from molfun.training.distributed import setup_ddp, cleanup_ddp

# Initialize process group
setup_ddp()

model = MolfunStructureModel.from_pretrained("openfold_v2", device=f"cuda:{local_rank}")
model.fit(
    train_dataset=ds,
    strategy="full",
    distributed="ddp",
    epochs=10,
)

cleanup_ddp()
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"nccl"` | Communication backend (`"nccl"`, `"gloo"`) |
| `find_unused_parameters` | `bool` | `False` | Enable if some parameters are not used every forward pass |
| `gradient_as_bucket_view` | `bool` | `True` | Memory optimization for gradient communication |

---

## FSDP -- Fully Sharded Data Parallel

Shards model parameters, gradients, and optimizer state across GPUs.
Enables training models that do not fit on a single GPU.

### Usage

```python
from molfun.training.distributed import setup_fsdp

model = MolfunStructureModel.from_pretrained("openfold_v2")
model.fit(
    train_dataset=ds,
    strategy="full",
    distributed="fsdp",
    fsdp_config={
        "sharding_strategy": "FULL_SHARD",
        "cpu_offload": False,
        "mixed_precision": True,
    },
    epochs=10,
)
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sharding_strategy` | `str` | `"FULL_SHARD"` | `"FULL_SHARD"`, `"SHARD_GRAD_OP"`, `"NO_SHARD"` |
| `cpu_offload` | `bool` | `False` | Offload parameters and gradients to CPU |
| `mixed_precision` | `bool` | `True` | Use mixed precision (fp16/bf16) for communication |
| `auto_wrap_policy` | `str \| None` | `None` | FSDP wrapping policy for sub-modules |
| `activation_checkpointing` | `bool` | `False` | Enable gradient checkpointing to save memory |

---

## Multi-Node Training

```bash
# Node 0 (master)
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=10.0.0.1 \
    --master_port=29500 \
    train.py

# Node 1
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=1 \
    --master_addr=10.0.0.1 \
    --master_port=29500 \
    train.py
```

## Recommendations

| Scenario | Recommended |
|----------|-------------|
| Model fits on 1 GPU, want faster training | DDP |
| Model does not fit on 1 GPU | FSDP (`FULL_SHARD`) |
| Moderate memory pressure | FSDP (`SHARD_GRAD_OP`) |
| Very large models + limited GPUs | FSDP + `cpu_offload` + `activation_checkpointing` |
