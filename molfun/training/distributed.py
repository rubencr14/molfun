"""
Distributed training strategies.

Provides ``BaseDistributedStrategy`` (ABC) and concrete implementations
for DDP and FSDP.  These strategies wrap the model and loaders for
multi-GPU training, and are passed to ``FinetuneStrategy.fit()`` via the
``distributed`` parameter.

Design
------
Follows the Strategy pattern: the distributed backend is decoupled from
the fine-tuning strategy (LoRA, Partial, Full).  Any combination works::

    strategy = LoRAFinetune(rank=8, lr_lora=2e-4)
    dist = DDPStrategy(backend="nccl")
    strategy.fit(model, train_loader, val_loader, epochs=20, distributed=dist)

The ``launch()`` helper handles ``torch.distributed`` initialisation and
``mp.spawn`` so users don't need to write boilerplate.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler


class BaseDistributedStrategy(ABC):
    """Interface for distributed training backends."""

    @abstractmethod
    def setup(self, rank: int, world_size: int) -> None:
        """Initialise process group for this rank."""

    @abstractmethod
    def wrap_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Wrap model for distributed training.  Returns wrapped module."""

    @abstractmethod
    def wrap_loader(self, loader: DataLoader, rank: int, world_size: int) -> DataLoader:
        """Replace sampler with a ``DistributedSampler``."""

    @abstractmethod
    def cleanup(self) -> None:
        """Destroy process group."""

    @property
    @abstractmethod
    def is_main_process(self) -> bool:
        """True on rank 0 — controls logging, checkpointing, etc."""

    @property
    @abstractmethod
    def local_rank(self) -> int:
        """Local GPU rank for this process."""

    def barrier(self) -> None:
        """Synchronise all processes."""
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


# ------------------------------------------------------------------
# DDP
# ------------------------------------------------------------------

class DDPStrategy(BaseDistributedStrategy):
    """
    Distributed Data Parallel — replicate model on each GPU, synchronise
    gradients via all-reduce after each backward pass.

    Best for: models that fit in a single GPU's memory (most protein ML
    models: OpenFold ~93M params, ESMFold ~700M on A100 80 GB).

    Usage::

        dist = DDPStrategy(backend="nccl")
        strategy.fit(model, train_loader, val_loader, distributed=dist)

    Or with the launcher::

        from molfun.training.distributed import launch

        def train_fn(rank, world_size, dist):
            model = ...
            strategy = LoRAFinetune(...)
            strategy.fit(model, train_loader, val_loader, distributed=dist)

        launch(train_fn, DDPStrategy(backend="nccl"), world_size=4)
    """

    def __init__(
        self,
        backend: str = "nccl",
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
        static_graph: bool = False,
    ) -> None:
        self._backend = backend
        self._find_unused = find_unused_parameters
        self._bucket_view = gradient_as_bucket_view
        self._static_graph = static_graph
        self._rank = 0
        self._world_size = 1

    def setup(self, rank: int, world_size: int) -> None:
        self._rank = rank
        self._world_size = world_size
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=self._backend,
                rank=rank,
                world_size=world_size,
            )
        torch.cuda.set_device(rank)

    def wrap_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        model = model.to(device)
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self._rank],
            output_device=self._rank,
            find_unused_parameters=self._find_unused,
            gradient_as_bucket_view=self._bucket_view,
            static_graph=self._static_graph,
        )

    def wrap_loader(self, loader: DataLoader, rank: int, world_size: int) -> DataLoader:
        sampler = DistributedSampler(
            loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last,
        )

    def cleanup(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @property
    def is_main_process(self) -> bool:
        return self._rank == 0

    @property
    def local_rank(self) -> int:
        return self._rank


# ------------------------------------------------------------------
# FSDP
# ------------------------------------------------------------------

class FSDPStrategy(BaseDistributedStrategy):
    """
    Fully Sharded Data Parallel — shard parameters, gradients, and
    optimizer state across GPUs.

    Best for: models too large for a single GPU, or when you need to
    maximise batch size per GPU.

    Args:
        backend: Process group backend (``nccl`` for GPU).
        sharding_strategy: ``"full"`` (ZeRO-3), ``"shard_grad_op"`` (ZeRO-2),
            ``"no_shard"`` (DDP-equivalent).
        mixed_precision: Enable bf16/fp16 compute.  ``"bf16"`` or ``"fp16"``.
        cpu_offload: Offload parameters to CPU between forward/backward.
        activation_checkpointing: Apply gradient checkpointing to wrapped
            modules automatically.
        auto_wrap_min_params: Minimum parameter count for FSDP auto-wrapping
            of submodules.  ``0`` wraps everything in one flat group.
    """

    def __init__(
        self,
        backend: str = "nccl",
        sharding_strategy: str = "full",
        mixed_precision: Optional[str] = None,
        cpu_offload: bool = False,
        activation_checkpointing: bool = False,
        auto_wrap_min_params: int = 100_000,
    ) -> None:
        self._backend = backend
        self._sharding = sharding_strategy
        self._mp_dtype = mixed_precision
        self._cpu_offload = cpu_offload
        self._act_ckpt = activation_checkpointing
        self._min_params = auto_wrap_min_params
        self._rank = 0
        self._world_size = 1

    def setup(self, rank: int, world_size: int) -> None:
        self._rank = rank
        self._world_size = world_size
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=self._backend,
                rank=rank,
                world_size=world_size,
            )
        torch.cuda.set_device(rank)

    def wrap_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            MixedPrecision,
            CPUOffload,
        )
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import functools

        sharding_map = {
            "full": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
        }
        sharding = sharding_map.get(self._sharding, ShardingStrategy.FULL_SHARD)

        mp_policy = None
        if self._mp_dtype == "bf16":
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif self._mp_dtype == "fp16":
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

        cpu_offload = CPUOffload(offload_params=True) if self._cpu_offload else None

        wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=self._min_params,
        )

        model = model.to(device)
        wrapped = FSDP(
            model,
            sharding_strategy=sharding,
            mixed_precision=mp_policy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=wrap_policy,
            device_id=self._rank,
        )

        if self._act_ckpt:
            _apply_activation_checkpointing(wrapped)

        return wrapped

    def wrap_loader(self, loader: DataLoader, rank: int, world_size: int) -> DataLoader:
        sampler = DistributedSampler(
            loader.dataset, num_replicas=world_size, rank=rank, shuffle=True,
        )
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last,
        )

    def cleanup(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @property
    def is_main_process(self) -> bool:
        return self._rank == 0

    @property
    def local_rank(self) -> int:
        return self._rank


# ------------------------------------------------------------------
# Launcher
# ------------------------------------------------------------------

def launch(
    fn,
    distributed: BaseDistributedStrategy,
    world_size: Optional[int] = None,
):
    """
    Launch a distributed training function across ``world_size`` processes.

    Each process calls ``fn(rank, world_size, distributed)``.

    Args:
        fn: Training function with signature ``(rank, world_size, dist) -> None``.
        distributed: The distributed strategy to use.
        world_size: Number of GPUs.  Defaults to ``torch.cuda.device_count()``.

    Usage::

        def train(rank, world_size, dist):
            dist.setup(rank, world_size)
            model = ...
            strategy = LoRAFinetune(...)
            strategy.fit(model, train_loader, distributed=dist)
            dist.cleanup()

        launch(train, DDPStrategy(), world_size=4)
    """
    if world_size is None:
        world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA GPUs available for distributed training.")

    if world_size == 1:
        distributed.setup(0, 1)
        fn(0, 1, distributed)
        distributed.cleanup()
    else:
        import torch.multiprocessing as mp
        mp.spawn(
            _worker,
            args=(world_size, fn, distributed),
            nprocs=world_size,
            join=True,
        )


def _worker(rank: int, world_size: int, fn, distributed: BaseDistributedStrategy):
    """Entry point for each spawned process."""
    distributed.setup(rank, world_size)
    try:
        fn(rank, world_size, distributed)
    finally:
        distributed.cleanup()


# ------------------------------------------------------------------
# Internal: activation checkpointing for FSDP
# ------------------------------------------------------------------

def _apply_activation_checkpointing(model: nn.Module) -> None:
    """
    Apply gradient checkpointing to FSDP-wrapped submodules.

    Targets common block types in protein ML models (Evoformer layers,
    Transformer blocks, etc.).
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )
    except ImportError:
        return

    block_types = set()
    for module in model.modules():
        name = type(module).__name__.lower()
        if any(k in name for k in ("block", "layer", "evoformer", "pairformer")):
            block_types.add(type(module))

    if block_types:
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=lambda m: type(m) in block_types,
        )
