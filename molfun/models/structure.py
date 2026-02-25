"""
Unified structure model for inference and fine-tuning.

Usage:

    # Inference
    model = MolfunStructureModel("openfold", config=cfg, weights="ckpt.pt")
    output = model.predict(batch)

    # Fine-tuning with strategy
    from molfun.training import LoRAFinetune

    model = MolfunStructureModel(
        "openfold", config=cfg, weights="ckpt.pt",
        head="affinity", head_config={"single_dim": 384},
    )
    strategy = LoRAFinetune(rank=8, lr_lora=1e-4, lr_head=1e-3)
    history = model.fit(train_loader, val_loader, strategy=strategy, epochs=10)
    model.save("checkpoint/")

    # Custom model from pluggable components
    from molfun.modules.builder import ModelBuilder

    built = ModelBuilder(
        embedder="input", block="pairformer", structure_module="ipa",
        n_blocks=8, block_config={"d_single": 256, "d_pair": 128},
    ).build()
    model = MolfunStructureModel.from_custom(built, head="affinity", head_config={"single_dim": 256})

    # Swap modules in a pre-trained model
    model = MolfunStructureModel("openfold", config=cfg, weights="ckpt.pt")
    model.swap("structure_module", MyCustomSM())
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from molfun.adapters.base import BaseAdapter
from molfun.peft.lora import MolfunPEFT
from molfun.heads.affinity import AffinityHead
from molfun.heads.structure import StructureLossHead
from molfun.core.types import TrunkOutput

if TYPE_CHECKING:
    from molfun.training.base import FinetuneStrategy


ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {}
HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "affinity": AffinityHead,
    "structure": StructureLossHead,
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
    interface for inference, fine-tuning, task heads, and checkpointing.

    The model itself is agnostic to the training strategy. Strategies
    (HeadOnly, LoRA, Partial, Full) are passed to ``fit()`` and handle
    all freezing, param groups, schedulers, EMA, etc.
    """

    def __init__(
        self,
        name: str,
        model: Optional[nn.Module] = None,
        config: Optional[object] = None,
        weights: Optional[str] = None,
        device: str = "cuda",
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
            head: Task head name ("affinity").
            head_config: Head kwargs (single_dim, hidden_dim, ...).
        """
        _register_adapters()

        self.name = name
        self.device = device

        adapter_cls = ADAPTER_REGISTRY.get(name)
        if adapter_cls is None:
            raise ValueError(
                f"Unknown model '{name}'. Available: {list(ADAPTER_REGISTRY)}"
            )
        self.adapter: BaseAdapter = adapter_cls(
            model=model, config=config, weights_path=weights, device=device,
        )

        self._peft: Optional[MolfunPEFT] = None
        self._strategy: Optional[FinetuneStrategy] = None

        self.head: Optional[nn.Module] = None
        if head:
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
        For StructureLossHead, "preds" is the scalar structure loss.
        """
        trunk_output = self.adapter(batch)
        result = {"trunk_output": trunk_output}
        if self.head is not None:
            result["preds"] = self.head(trunk_output, mask=mask, batch=batch)
        return result

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        strategy: Optional[FinetuneStrategy] = None,
        epochs: int = 10,
    ) -> list[dict]:
        """
        Fine-tune the model using the given strategy.

        Args:
            train_loader: Training data.
            val_loader: Validation data (optional).
            strategy: FinetuneStrategy instance (HeadOnly, LoRA, Partial, Full).
            epochs: Number of training epochs.

        Returns:
            List of per-epoch metric dicts.
        """
        if strategy is not None:
            self._strategy = strategy
        if self._strategy is None:
            raise RuntimeError(
                "No strategy provided. Pass e.g. strategy=HeadOnlyFinetune(lr=1e-3)"
            )
        if self.head is None:
            raise RuntimeError(
                "No head configured. Pass head='affinity' or head='structure'."
            )
        return self._strategy.fit(self, train_loader, val_loader, epochs)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save PEFT adapters + head weights."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = {"name": self.name}
        if self._strategy is not None:
            meta["strategy"] = self._strategy.describe()
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
        info = {"name": self.name, "device": self.device}
        info["adapter"] = self.adapter.param_summary()
        if self._peft is not None:
            info["peft"] = self._peft.summary()
        if self._strategy is not None:
            info["strategy"] = self._strategy.describe()
        if self.head is not None:
            info["head"] = {
                "type": type(self.head).__name__,
                "params": sum(p.numel() for p in self.head.parameters()),
            }
        return info

    # ------------------------------------------------------------------
    # Custom model building
    # ------------------------------------------------------------------

    @classmethod
    def from_custom(
        cls,
        adapter: BaseAdapter,
        device: str = "cuda",
        head: Optional[str] = None,
        head_config: Optional[dict] = None,
    ) -> "MolfunStructureModel":
        """
        Create a MolfunStructureModel from a custom adapter (e.g. BuiltModel).

        This bypasses the ADAPTER_REGISTRY and directly uses the provided
        adapter, enabling custom architectures built with ModelBuilder
        or hand-crafted nn.Modules that implement BaseAdapter.

        Usage::

            from molfun.modules.builder import ModelBuilder
            built = ModelBuilder(
                embedder="input", block="pairformer", structure_module="ipa",
            ).build()
            model = MolfunStructureModel.from_custom(
                built, head="affinity", head_config={"single_dim": 256},
            )
        """
        instance = cls.__new__(cls)
        instance.name = "custom"
        instance.device = device
        instance.adapter = adapter.to(device)
        instance._peft = None
        instance._strategy = None
        instance.head = None
        if head:
            head_cls = HEAD_REGISTRY.get(head)
            if head_cls is None:
                raise ValueError(f"Unknown head: {head}. Available: {list(HEAD_REGISTRY)}")
            instance.head = head_cls(**(head_config or {})).to(device)
        return instance

    # ------------------------------------------------------------------
    # Module swapping
    # ------------------------------------------------------------------

    def swap(
        self,
        target_path: str,
        new_module: nn.Module,
        transfer_weights: bool = False,
    ) -> nn.Module:
        """
        Replace an internal submodule of the adapter's model.

        Uses ModuleSwapper under the hood. The target_path is relative
        to the adapter's internal model (e.g. "structure_module",
        "evoformer.blocks.0").

        Args:
            target_path: Dotted path to the submodule.
            new_module: Replacement module.
            transfer_weights: Copy matching weights from old module.

        Returns:
            The old (replaced) module.

        Usage::

            from molfun.modules.structure_module import DiffusionStructureModule
            model.swap("structure_module", DiffusionStructureModule(...))
        """
        from molfun.modules.swapper import ModuleSwapper
        target = self.adapter if hasattr(self.adapter, 'model') else self.adapter
        actual_model = getattr(target, 'model', target)
        return ModuleSwapper.swap(
            actual_model, target_path, new_module,
            transfer_weights=transfer_weights,
        )

    def swap_all(
        self,
        pattern: str,
        factory,
        transfer_weights: bool = False,
    ) -> int:
        """
        Swap all submodules matching a regex pattern.

        Args:
            pattern: Regex pattern for module names.
            factory: ``factory(name, old_module) → new_module``.
            transfer_weights: Copy matching weights from old modules.

        Returns:
            Number of modules swapped.
        """
        from molfun.modules.swapper import ModuleSwapper
        target = self.adapter if hasattr(self.adapter, 'model') else self.adapter
        actual_model = getattr(target, 'model', target)
        return ModuleSwapper.swap_all(
            actual_model, pattern, factory,
            transfer_weights=transfer_weights,
        )

    def discover_modules(self, pattern: Optional[str] = None) -> list[tuple[str, nn.Module]]:
        """List swappable modules inside the model."""
        from molfun.modules.swapper import ModuleSwapper
        target = self.adapter if hasattr(self.adapter, 'model') else self.adapter
        actual_model = getattr(target, 'model', target)
        return ModuleSwapper.discover(actual_model, pattern=pattern)

    # ------------------------------------------------------------------
    # Hugging Face Hub
    # ------------------------------------------------------------------

    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        metrics: Optional[dict] = None,
        dataset_name: Optional[str] = None,
        commit_message: str = "Upload Molfun model",
    ) -> str:
        """
        Push model checkpoint + auto-generated model card to HF Hub.

        Args:
            repo_id: Hub repo (e.g. "user/my-affinity-model").
            token: HF API token (or set HF_TOKEN env var).
            private: Whether the repo should be private.
            metrics: Evaluation metrics to include in the model card.
            dataset_name: Training dataset name for the card.
            commit_message: Git commit message for the upload.

        Returns:
            URL of the uploaded repo.

        Usage::

            model.push_to_hub("rubencr/kinase-affinity-lora", metrics={"mae": 0.42})
        """
        import tempfile

        from molfun.tracking.hf_tracker import HuggingFaceTracker
        from molfun.tracking.model_card import generate_model_card

        tracker = HuggingFaceTracker(
            repo_id=repo_id, token=token, private=private,
        )

        with tempfile.TemporaryDirectory() as tmp:
            self.save(tmp)
            tracker.start_run(name=commit_message, config=self.summary())
            if metrics:
                tracker.log_metrics(metrics)
            tracker.log_artifact(tmp, name="checkpoint")

        card = generate_model_card(
            model_summary=self.summary(),
            metrics=metrics,
            dataset_name=dataset_name,
        )
        tracker.upload_model_card(card)
        tracker.end_run()

        return f"https://huggingface.co/{repo_id}"

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        device: str = "cpu",
        head: Optional[str] = None,
        head_config: Optional[dict] = None,
    ) -> "MolfunStructureModel":
        """
        Download and load a model from Hugging Face Hub.

        Args:
            repo_id: Hub repo (e.g. "user/my-affinity-model").
            token: HF API token.
            revision: Git revision (branch, tag, commit hash).
            device: Target device.
            head: Task head to attach (overrides saved head).
            head_config: Head kwargs (overrides saved config).

        Returns:
            Loaded MolfunStructureModel.

        Usage::

            model = MolfunStructureModel.from_hub("rubencr/kinase-affinity-lora")
            output = model.predict(batch)
        """
        from molfun.tracking.hf_tracker import HuggingFaceTracker

        tracker = HuggingFaceTracker(repo_id=repo_id, token=token)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = tracker.download_repo(tmp, revision=revision)
            ckpt_dir = Path(local_dir) / "checkpoint"

            if not ckpt_dir.exists():
                ckpt_dir = Path(local_dir)

            meta_path = ckpt_dir / "meta.pt"
            if meta_path.exists():
                meta = torch.load(meta_path, map_location=device, weights_only=True)
                name = meta.get("name", "openfold")
            else:
                name = "custom"

            if name == "custom":
                raise ValueError(
                    "Cannot reconstruct custom models from Hub yet. "
                    "Use from_custom() with a ModelBuilder config instead."
                )

            instance = cls(
                name=name, device=device,
                head=head, head_config=head_config,
            )
            instance.load(str(ckpt_dir))

        return instance

    # ------------------------------------------------------------------
    # Registry queries
    # ------------------------------------------------------------------

    @staticmethod
    def available_models() -> list[str]:
        _register_adapters()
        return list(ADAPTER_REGISTRY.keys())

    @staticmethod
    def available_heads() -> list[str]:
        return list(HEAD_REGISTRY.keys())
