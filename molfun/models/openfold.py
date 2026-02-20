"""
High-level OpenFold wrapper for inference and fine-tuning.

Usage:

    # Inference
    model = OpenFold(config=openfold_config, weights="path/to/weights.pt")
    output = model.predict(batch)

    # Fine-tuning with LoRA + affinity head
    model = OpenFold(
        config=openfold_config,
        weights="path/to/weights.pt",
        fine_tune=True,
        peft="lora",
        peft_config={"rank": 8, "target_modules": ["linear_q", "linear_v"]},
        head="affinity",
        head_config={"single_dim": 384, "hidden_dim": 256},
    )
    model.fit(train_loader, val_loader, epochs=10, lr=1e-4)
    model.save("my_finetuned/")
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from molfun.adapters.openfold import OpenFoldAdapter
from molfun.peft.lora import MolfunPEFT
from molfun.heads.affinity import AffinityHead
from molfun.core.types import TrunkOutput

HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "affinity": AffinityHead,
}


class OpenFold:
    """
    High-level OpenFold API for inference and fine-tuning.
    
    Composes: OpenFoldAdapter (model wrapper) + MolfunPEFT (LoRA/IA³) + Head (task).
    """

    def __init__(
        self,
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
            model: Pre-built OpenFold nn.Module. If None, built from config.
            config: OpenFold config object. Required if model is None.
            weights: Path to model checkpoint.
            device: Target device.
            fine_tune: If True, freeze trunk and set up PEFT + head.
            peft: PEFT method ("lora" or "ia3"). Ignored if fine_tune=False.
            peft_config: PEFT kwargs (rank, alpha, target_modules, etc.).
            head: Task head ("affinity"). Ignored if fine_tune=False.
            head_config: Head kwargs (single_dim, hidden_dim, etc.).
        """
        self.device = device
        self.fine_tune = fine_tune

        # 1. Adapter wraps the raw OpenFold model
        self.adapter = OpenFoldAdapter(
            model=model, config=config, weights_path=weights, device=device,
        )

        # 2. PEFT (optional, only for fine-tuning)
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
            self._peft.apply(self.adapter.model.evoformer)

        elif fine_tune:
            self.adapter.freeze_trunk()

        # 3. Task head (optional, only for fine-tuning)
        self.head: Optional[nn.Module] = None
        if fine_tune and head:
            head_cls = HEAD_REGISTRY.get(head)
            if head_cls is None:
                raise ValueError(f"Unknown head: {head}. Available: {list(HEAD_REGISTRY)}")
            head_config = head_config or {}
            self.head = head_cls(**head_config).to(device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, batch: dict) -> TrunkOutput:
        """Run inference and return normalized output."""
        self.adapter.eval()
        return self.adapter(batch)

    def forward(self, batch: dict, mask: Optional[torch.Tensor] = None) -> dict:
        """
        Full forward pass: adapter → head (if fine-tuning).
        
        Returns:
            Dict with "trunk_output" and optionally "preds".
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

        Args:
            train_loader: DataLoader yielding (batch_dict, targets) or dicts with "labels".
            val_loader: Optional validation DataLoader.
            epochs: Number of training epochs.
            lr: Learning rate.
            weight_decay: AdamW weight decay.
            grad_clip: Max gradient norm (0 to disable).
            amp: Use automatic mixed precision.
            loss_fn: Loss function for head ("mse" or "huber").
            callbacks: Optional list of callback objects with on_epoch_end(epoch, metrics).
            
        Returns:
            List of per-epoch metrics dicts.
        """
        if self.head is None:
            raise RuntimeError("No head configured. Pass head='affinity' to enable training.")

        # Collect trainable parameters
        params = list(self.head.parameters())
        if self._peft is not None:
            params += self._peft.trainable_parameters()

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler(enabled=amp)
        history = []

        for epoch in range(epochs):
            # Train
            self.adapter.train()
            self.head.train()
            train_metrics = self._train_epoch(
                train_loader, optimizer, scaler, grad_clip, amp, loss_fn,
            )

            # Validate
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._val_epoch(val_loader, loss_fn)

            metrics = {"epoch": epoch + 1, **train_metrics, **val_metrics}
            history.append(metrics)

            if callbacks:
                for cb in callbacks:
                    cb.on_epoch_end(epoch, metrics)

        return history

    def _train_epoch(
        self, loader: DataLoader, optimizer, scaler, grad_clip: float, amp: bool, loss_fn: str,
    ) -> dict:
        total_loss = 0.0
        n_batches = 0

        for batch_data in loader:
            batch, targets, mask = self._unpack_batch(batch_data)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=amp):
                result = self.forward(batch, mask=mask)
                losses = self.head.loss(result["preds"], targets, loss_fn=loss_fn)
                loss = losses["affinity_loss"]

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                params = list(self.head.parameters())
                if self._peft:
                    params += self._peft.trainable_parameters()
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return {"train_loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, loss_fn: str) -> dict:
        self.adapter.eval()
        self.head.eval()
        total_loss = 0.0
        n_batches = 0

        for batch_data in loader:
            batch, targets, mask = self._unpack_batch(batch_data)
            result = self.forward(batch, mask=mask)
            losses = self.head.loss(result["preds"], targets, loss_fn=loss_fn)
            total_loss += losses["affinity_loss"].item()
            n_batches += 1

        return {"val_loss": total_loss / max(n_batches, 1)}

    @staticmethod
    def _unpack_batch(batch_data) -> tuple[dict, torch.Tensor, Optional[torch.Tensor]]:
        """
        Unpack DataLoader output into (features, targets, mask).

        Supports:
        - (features_dict, targets)         — from StructureDataset/AffinityDataset
        - (features_dict, targets, mask)   — explicit mask
        - dict with "labels" key           — self-contained batch
        """
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
        """Save fine-tuned weights (PEFT adapters + head)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._peft is not None:
            self._peft.save(str(path / "peft_adapter"))

        if self.head is not None:
            torch.save(self.head.state_dict(), path / "head.pt")

    def load(self, path: str) -> None:
        """Load fine-tuned weights (PEFT adapters + head)."""
        path = Path(path)

        peft_path = path / "peft_adapter"
        if self._peft is not None and peft_path.exists():
            self._peft.load(str(peft_path))

        head_path = path / "head.pt"
        if self.head is not None and head_path.exists():
            self.head.load_state_dict(
                torch.load(head_path, map_location=self.device, weights_only=True)
            )

    # ------------------------------------------------------------------
    # Merge (for production export)
    # ------------------------------------------------------------------

    def merge(self) -> None:
        """Merge PEFT weights into base model (removes adapter overhead)."""
        if self._peft is not None:
            self._peft.merge()

    def unmerge(self) -> None:
        """Undo merge."""
        if self._peft is not None:
            self._peft.unmerge()

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return model summary with parameter counts."""
        info = {"device": self.device, "fine_tune": self.fine_tune}
        info["adapter"] = self.adapter.param_summary()

        if self._peft is not None:
            info["peft"] = self._peft.summary()

        if self.head is not None:
            info["head"] = {
                "type": type(self.head).__name__,
                "params": sum(p.numel() for p in self.head.parameters()),
            }

        return info
