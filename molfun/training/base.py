"""
Base fine-tuning strategy with training infrastructure.

All fine-tuning modes share: warmup scheduler, EMA, gradient accumulation,
early stopping, AMP, grad clipping, and metrics tracking.
Subclasses only define *what to freeze* and *how to group parameters*.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from molfun.losses import LOSS_REGISTRY
from molfun.helpers.training import EMA, build_scheduler, unpack_batch, to_device


class FinetuneStrategy(ABC):
    """
    Base class for all fine-tuning strategies.

    Subclasses implement `setup()` (freeze/unfreeze logic) and
    `param_groups()` (optimizer parameter groups with per-group LR).

    The base provides the full training loop with:
    - Linear warmup + cosine/linear/constant scheduler
    - Exponential Moving Average (EMA)
    - Gradient accumulation
    - Mixed precision (AMP)
    - Gradient clipping
    - Early stopping on val_loss
    """

    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        scheduler: str = "cosine",
        min_lr: float = 1e-6,
        ema_decay: float = 0.0,
        grad_clip: float = 1.0,
        accumulation_steps: int = 1,
        amp: bool = True,
        early_stopping_patience: int = 0,
        loss_fn: str = "mse",
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.scheduler_name = scheduler
        self.min_lr = min_lr
        self.ema_decay = ema_decay
        self.grad_clip = grad_clip
        self.accumulation_steps = accumulation_steps
        self.amp = amp
        self.patience = early_stopping_patience
        self.loss_fn = loss_fn

        self._ema: Optional[EMA] = None
        self._setup_done = False

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def setup(self, model) -> None:
        """
        Configure the model for this strategy: freeze/unfreeze params,
        inject PEFT layers, attach heads, etc. Called once before training.
        Idempotent: calling multiple times has no effect.
        """
        if self._setup_done:
            return
        self._setup_impl(model)
        model._strategy = self
        self._setup_done = True

    @abstractmethod
    def _setup_impl(self, model) -> None:
        """Subclass override: freeze/unfreeze/inject logic."""

    @abstractmethod
    def param_groups(self, model) -> list[dict]:
        """
        Return optimizer parameter groups.

        Example: [
            {"params": head_params, "lr": 1e-3},
            {"params": lora_params, "lr": 1e-4},
        ]
        """

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True,
        tracker=None,
        distributed=None,
        gradient_checkpointing: bool = False,
    ) -> list[dict]:
        """
        Run the full training loop.

        Args:
            model: MolfunStructureModel instance.
            train_loader: Training data.
            val_loader: Optional validation data.
            epochs: Number of epochs.
            tracker: Optional BaseTracker for experiment logging.
            distributed: Optional ``BaseDistributedStrategy`` (DDPStrategy
                or FSDPStrategy) for multi-GPU training.
            gradient_checkpointing: Enable activation checkpointing to
                reduce peak VRAM (~40-60% savings, ~25-35% slower).

        Returns:
            List of per-epoch metric dicts.
        """
        self.setup(model)

        if gradient_checkpointing:
            from molfun.training.checkpointing import apply_gradient_checkpointing
            n_ckpt = apply_gradient_checkpointing(model.adapter)
            if verbose:
                print(f"  Gradient checkpointing: {n_ckpt} modules wrapped")

        if distributed is not None:
            device = torch.device(f"cuda:{distributed.local_rank}")
            model.adapter = distributed.wrap_model(model.adapter, device)
            train_loader = distributed.wrap_loader(
                train_loader, distributed.local_rank,
                distributed._world_size if hasattr(distributed, '_world_size') else 1,
            )
            if val_loader is not None:
                val_loader = distributed.wrap_loader(
                    val_loader, distributed.local_rank,
                    distributed._world_size if hasattr(distributed, '_world_size') else 1,
                )
            model.device = str(device)

        self._distributed = distributed

        groups = self.param_groups(model)
        optimizer = torch.optim.AdamW(groups, weight_decay=self.weight_decay)

        steps_per_epoch = len(train_loader) // self.accumulation_steps
        total_steps = steps_per_epoch * epochs
        scheduler = build_scheduler(
            optimizer, self.scheduler_name,
            self.warmup_steps, total_steps, self.min_lr,
        )

        scaler = torch.amp.GradScaler(enabled=self.amp)

        all_params = [p for g in groups for p in g["params"]]
        if self.ema_decay > 0:
            self._ema = EMA(all_params, decay=self.ema_decay)

        is_main = self._distributed is None or self._distributed.is_main_process

        if tracker is not None and is_main:
            tracker.log_config(self.describe())

        best_val = float("inf")
        patience_counter = 0
        history: list[dict] = []
        global_step = 0

        for epoch in range(epochs):
            # Set epoch on DistributedSampler for proper shuffling
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            model.adapter.train()
            if model.head is not None:
                model.head.train()

            train_loss, global_step = self._train_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                all_params, global_step,
            )

            metrics: dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }

            if val_loader is not None:
                val_metrics = self._val_epoch(model, val_loader)
                metrics.update(val_metrics)

                val_loss = val_metrics.get("val_loss", float("inf"))
                if self.patience > 0:
                    if val_loss < best_val:
                        best_val = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    metrics["patience"] = patience_counter

            history.append(metrics)

            if tracker is not None and is_main:
                tracker.log_metrics(metrics, step=epoch + 1)

            if verbose and is_main:
                val_str = f"{metrics['val_loss']:.4f}" if "val_loss" in metrics else "   —  "
                print(
                    f"  Epoch {epoch+1:>3}/{epochs}  "
                    f"train={train_loss:.4f}  val={val_str}  "
                    f"lr={metrics['lr']:.2e}",
                    flush=True,
                )

            if self.patience > 0 and patience_counter >= self.patience:
                if is_main:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        return history

    def _compute_loss(
        self,
        model,
        preds,
        targets: Optional[torch.Tensor],
        batch: dict,
    ) -> dict[str, torch.Tensor]:
        """
        Resolve and call the correct loss function.

        Two execution paths:

        **Structure heads** (e.g. StructureLossHead):
            ``head.forward()`` already called ``OpenFoldLoss`` internally and
            returned a scalar loss tensor.  ``head.loss(scalar, None)`` just
            wraps it in a dict.  The registry is used *inside* the head.

        **Affinity heads** (e.g. AffinityHead):
            ``head.forward()`` returns raw predictions.  ``head.loss()``
            delegates to ``LOSS_REGISTRY[loss_fn]`` to compute the actual
            loss.

        In both cases the training loop calls ``model.head.loss()``.
        The registry is what ``head.loss()`` uses internally — giving users
        one consistent place to register custom losses.
        """
        return model.head.loss(preds, targets, loss_fn=self.loss_fn)

    def _train_epoch(
        self, model, loader, optimizer, scheduler, scaler,
        all_params, global_step,
    ) -> tuple[float, int]:
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for i, batch_data in enumerate(loader):
            batch, targets, mask = unpack_batch(batch_data)
            batch = to_device(batch, model.device)
            if targets is not None:
                targets = targets.to(model.device)
            if mask is not None:
                mask = mask.to(model.device)

            with torch.amp.autocast("cuda", enabled=self.amp):
                result = model.forward(batch, mask=mask)
                losses = self._compute_loss(model, result["preds"], targets, batch)
                loss = next(iter(losses.values())) / self.accumulation_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * self.accumulation_steps
            n_batches += 1

            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(loader):
                if self.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if self._ema is not None:
                    self._ema.update()

        return total_loss / max(n_batches, 1), global_step

    @torch.no_grad()
    def _val_epoch(self, model, loader) -> dict:
        model.adapter.eval()
        if model.head is not None:
            model.head.eval()

        use_ema = self._ema is not None
        if use_ema:
            self._ema.apply()

        total_loss = 0.0
        n = 0
        for batch_data in loader:
            batch, targets, mask = unpack_batch(batch_data)
            batch = to_device(batch, model.device)
            if targets is not None:
                targets = targets.to(model.device)
            if mask is not None:
                mask = mask.to(model.device)
            result = model.forward(batch, mask=mask)
            losses = self._compute_loss(model, result["preds"], targets, batch)
            total_loss += next(iter(losses.values())).item()
            n += 1

        if use_ema:
            self._ema.restore()

        return {"val_loss": total_loss / max(n, 1)}

    # ------------------------------------------------------------------
    # EMA access
    # ------------------------------------------------------------------

    def apply_ema(self, model) -> None:
        """Copy EMA weights into the model permanently (for export/inference)."""
        if self._ema is not None:
            self._ema.apply()
            self._ema = None

    @property
    def ema(self) -> Optional[EMA]:
        return self._ema

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def describe(self) -> dict:
        return {
            "strategy": type(self).__name__,
            "lr": self.lr,
            "scheduler": self.scheduler_name,
            "warmup_steps": self.warmup_steps,
            "ema_decay": self.ema_decay,
            "accumulation_steps": self.accumulation_steps,
            "grad_clip": self.grad_clip,
            "amp": self.amp,
            "early_stopping_patience": self.patience,
        }

