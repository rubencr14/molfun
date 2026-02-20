"""
Unified structure model for inference and fine-tuning.

Usage:

    # Inference with OpenFold
    model = MolfunStructureModel("openfold", config=cfg, weights="ckpt.pt")
    output = model.predict(batch)

    # Fine-tuning with LoRA + affinity head
    model = MolfunStructureModel(
        "openfold",
        config=cfg,
        weights="ckpt.pt",
        fine_tune=True,
        peft="lora",
        peft_config={"rank": 8},
        head="affinity",
        head_config={"single_dim": 384},
    )
    history = model.fit(train_loader, val_loader, epochs=10)
    model.save("checkpoint/")

    # Future models use the same API
    model = MolfunStructureModel("esmfold", ...)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from molfun.adapters.base import BaseAdapter
from molfun.peft.lora import MolfunPEFT
from molfun.heads.affinity import AffinityHead
from molfun.core.types import TrunkOutput


ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {}
HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "affinity": AffinityHead,
}


def _register_adapters():
    """Lazy registration to avoid import errors for uninstalled backends."""
    if ADAPTER_REGISTRY:
        return
    from molfun.adapters.openfold import OpenFoldAdapter
    ADAPTER_REGISTRY["openfold"] = OpenFoldAdapter


class MolfunStructureModel:
    """
    Unified API for protein structure models.

    Wraps any registered adapter (OpenFold, ESMFold, ...) with a common
    interface for inference, fine-tuning (PEFT), task heads, and checkpointing.
    """

    def __init__(
        self,
        name: str,
        model: Optional[nn.Module] = None,
        config: Optional[object] = None,
        weights: Optional[str] = None,
        device: str = "cuda",
        fine_tune: bool = False,
        peft: Optional[str] = None,
        peft_config: Optional[dict] = None,
        head: Optional[str] = None,
        head_config: Optional[dict] = None,
    ):
        """
        Args:
            name: Model backend ("openfold", "esmfold", ...).
            model: Pre-built nn.Module. If None, built from config.
            config: Backend-specific config object.
            weights: Path to model checkpoint.
            device: Target device.
            fine_tune: Freeze trunk and enable PEFT + head training.
            peft: PEFT method ("lora" or "ia3"). Requires fine_tune=True.
            peft_config: PEFT kwargs (rank, alpha, target_modules, ...).
            head: Task head name ("affinity"). Requires fine_tune=True.
            head_config: Head kwargs (single_dim, hidden_dim, ...).
        """
        _register_adapters()

        self.name = name
        self.device = device
        self.fine_tune = fine_tune

        # 1. Adapter
        adapter_cls = ADAPTER_REGISTRY.get(name)
        if adapter_cls is None:
            raise ValueError(
                f"Unknown model '{name}'. Available: {list(ADAPTER_REGISTRY)}"
            )
        self.adapter: BaseAdapter = adapter_cls(
            model=model, config=config, weights_path=weights, device=device,
        )

        # 2. PEFT
        self._peft: Optional[MolfunPEFT] = None
        if fine_tune and peft:
            peft_config = peft_config or {}
            if peft == "lora":
                self._peft = MolfunPEFT.lora(**peft_config)
            elif peft == "ia3":
                self._peft = MolfunPEFT.ia3(**peft_config)
            else:
                raise ValueError(f"Unknown PEFT method: {peft}. Use 'lora' or 'ia3'.")
            self.adapter.freeze_trunk()
            self._peft.apply(self.adapter.peft_target_module)
        elif fine_tune:
            self.adapter.freeze_trunk()

        # 3. Head
        self.head: Optional[nn.Module] = None
        if fine_tune and head:
            head_cls = HEAD_REGISTRY.get(head)
            if head_cls is None:
                raise ValueError(f"Unknown head: {head}. Available: {list(HEAD_REGISTRY)}")
            self.head = head_cls(**(head_config or {})).to(device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, batch: dict) -> TrunkOutput:
        """Run inference (no grad, eval mode)."""
        self.adapter.eval()
        return self.adapter(batch)

    def forward(self, batch: dict, mask: Optional[torch.Tensor] = None) -> dict:
        """
        Full forward: adapter → head.

        Returns dict with "trunk_output" and optionally "preds".
        """
        trunk_output = self.adapter(batch)
        result = {"trunk_output": trunk_output}
        if self.head is not None:
            result["preds"] = self.head(trunk_output, mask=mask)
        return result

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        amp: bool = True,
        loss_fn: str = "mse",
        callbacks: Optional[list] = None,
    ) -> list[dict]:
        """
        Fine-tune the model.

        Returns list of per-epoch metrics dicts.
        """
        if self.head is None:
            raise RuntimeError("No head configured. Pass head='affinity' to enable training.")

        params = list(self.head.parameters())
        if self._peft is not None:
            params += self._peft.trainable_parameters()

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler(enabled=amp)
        history = []

        for epoch in range(epochs):
            self.adapter.train()
            self.head.train()
            train_metrics = self._train_epoch(train_loader, optimizer, scaler, grad_clip, amp, loss_fn)

            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._val_epoch(val_loader, loss_fn)

            metrics = {"epoch": epoch + 1, **train_metrics, **val_metrics}
            history.append(metrics)

            if callbacks:
                for cb in callbacks:
                    cb.on_epoch_end(epoch, metrics)

        return history

    def _train_epoch(self, loader, optimizer, scaler, grad_clip, amp, loss_fn) -> dict:
        total_loss = 0.0
        n = 0
        for batch_data in loader:
            batch, targets, mask = self._unpack_batch(batch_data)
            batch = _to_device(batch, self.device)
            targets = targets.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                result = self.forward(batch, mask=mask)
                losses = self.head.loss(result["preds"], targets, loss_fn=loss_fn)
                loss = losses["affinity_loss"]
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                all_params = list(self.head.parameters())
                if self._peft:
                    all_params += self._peft.trainable_parameters()
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n += 1
        return {"train_loss": total_loss / max(n, 1)}

    @torch.no_grad()
    def _val_epoch(self, loader, loss_fn) -> dict:
        self.adapter.eval()
        self.head.eval()
        total_loss = 0.0
        n = 0
        for batch_data in loader:
            batch, targets, mask = self._unpack_batch(batch_data)
            batch = _to_device(batch, self.device)
            targets = targets.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)
            result = self.forward(batch, mask=mask)
            losses = self.head.loss(result["preds"], targets, loss_fn=loss_fn)
            total_loss += losses["affinity_loss"].item()
            n += 1
        return {"val_loss": total_loss / max(n, 1)}

    @staticmethod
    def _unpack_batch(batch_data) -> tuple[dict, torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(batch_data, (list, tuple)):
            if len(batch_data) == 3:
                return batch_data[0], batch_data[1], batch_data[2]
            features, targets = batch_data[0], batch_data[1]
            mask = features.get("all_atom_mask") if isinstance(features, dict) else None
            return features, targets, mask
        if isinstance(batch_data, dict):
            targets = batch_data.pop("labels")
            mask = batch_data.get("all_atom_mask") or batch_data.get("mask")
            return batch_data, targets, mask
        raise ValueError(f"Unsupported batch format: {type(batch_data)}")

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save PEFT adapters + head weights."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = {"name": self.name, "fine_tune": self.fine_tune}
        torch.save(meta, path / "meta.pt")

        if self._peft is not None:
            self._peft.save(str(path / "peft_adapter"))
        if self.head is not None:
            torch.save(self.head.state_dict(), path / "head.pt")

    def load(self, path: str) -> None:
        """Load PEFT adapters + head weights."""
        path = Path(path)
        if self._peft is not None and (path / "peft_adapter").exists():
            self._peft.load(str(path / "peft_adapter"))
        if self.head is not None and (path / "head.pt").exists():
            self.head.load_state_dict(
                torch.load(path / "head.pt", map_location=self.device, weights_only=True)
            )

    # ------------------------------------------------------------------
    # Merge / info
    # ------------------------------------------------------------------

    def merge(self) -> None:
        """Merge PEFT weights into base model for production export."""
        if self._peft is not None:
            self._peft.merge()

    def unmerge(self) -> None:
        if self._peft is not None:
            self._peft.unmerge()

    def summary(self) -> dict:
        info = {"name": self.name, "device": self.device, "fine_tune": self.fine_tune}
        info["adapter"] = self.adapter.param_summary()
        if self._peft is not None:
            info["peft"] = self._peft.summary()
        if self.head is not None:
            info["head"] = {
                "type": type(self.head).__name__,
                "params": sum(p.numel() for p in self.head.parameters()),
            }
        return info

    @staticmethod
    def available_models() -> list[str]:
        _register_adapters()
        return list(ADAPTER_REGISTRY.keys())

    @staticmethod
    def available_heads() -> list[str]:
        return list(HEAD_REGISTRY.keys())


def _to_device(batch: dict, device: str) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
