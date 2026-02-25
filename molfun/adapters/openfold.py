"""OpenFold (AlphaFold2) adapter for Molfun."""

from __future__ import annotations
from typing import Optional
import re
import torch
import torch.nn as nn

from molfun.adapters.base import BaseAdapter
from molfun.core.types import TrunkOutput


class OpenFoldAdapter(BaseAdapter):
    """
    Wraps an OpenFold AlphaFold model behind a normalized API.
    
    Exposes:
    - forward() → TrunkOutput with single/pair representations and structure
    - peft_target_module → evoformer (for PEFT injection)
    - freeze_trunk() / unfreeze_trunk() for fine-tuning control
    """

    EVOFORMER_KEY = "evoformer"
    STRUCTURE_MODULE_KEY = "structure_module"
    INPUT_EMBEDDER_KEY = "input_embedder"

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[object] = None,
        weights_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Args:
            model: Pre-built OpenFold AlphaFold model. If None, built from config.
            config: OpenFold model config (mlc_config style). Used only if model is None.
            weights_path: Path to checkpoint. Used only if model is None.
            device: Target device.
        """
        super().__init__()
        self.device = device

        if model is not None:
            self.model = model
        elif config is not None:
            self.model = self._build_from_config(config, weights_path)
        else:
            raise ValueError("Provide either `model` or `config`.")

        self.model.to(device)

    @staticmethod
    def _build_from_config(config, weights_path: Optional[str] = None) -> nn.Module:
        try:
            from openfold.model.model import AlphaFold
        except ImportError:
            raise ImportError(
                "OpenFold is required: pip install openfold or install from "
                "https://github.com/aqlaboratory/openfold"
            )

        model = AlphaFold(config)
        if weights_path:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            state = _remap_openfold_keys(state, model)
            model.load_state_dict(state, strict=False)
        return model

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> TrunkOutput:
        """
        Run OpenFold forward and return normalized TrunkOutput.

        Args:
            batch: OpenFold-style feature dict (see AlphaFold.forward docs).

        Returns:
            TrunkOutput with single_repr, pair_repr, structure_coords, confidence.
        """
        outputs = self.model(batch)

        # pLDDT from structure module logits
        confidence = None
        if "plddt" in outputs:
            confidence = outputs["plddt"]
        elif "sm" in outputs and "plddt" not in outputs:
            try:
                from openfold.utils.loss import compute_plddt
                confidence = compute_plddt(outputs["sm"]["single"])
            except (ImportError, KeyError):
                pass

        return TrunkOutput(
            single_repr=outputs["single"],
            pair_repr=outputs.get("pair"),
            structure_coords=outputs.get("final_atom_positions"),
            confidence=confidence,
            extra={
                "msa": outputs.get("msa"),
                "sm": outputs.get("sm"),
                "num_recycles": outputs.get("num_recycles"),
                # Full output dict required by StructureLossHead / AlphaFoldLoss
                "_raw_outputs": outputs,
            },
        )

    # ------------------------------------------------------------------
    # PEFT targeting
    # ------------------------------------------------------------------
    def get_targetable_modules(self) -> dict[str, nn.Module]:
        """
        Return named modules suitable for PEFT injection.

        Returns dict with keys like:
            "evoformer" → EvoformerStack
            "evoformer.blocks.0" → first Evoformer block
            "structure_module" → StructureModule
            "input_embedder" → InputEmbedder
        """
        modules = {}
        for key in (self.EVOFORMER_KEY, self.STRUCTURE_MODULE_KEY, self.INPUT_EMBEDDER_KEY):
            mod = getattr(self.model, key, None)
            if mod is not None:
                modules[key] = mod

        # Expose individual evoformer blocks
        evoformer = getattr(self.model, self.EVOFORMER_KEY, None)
        if evoformer is not None:
            blocks = getattr(evoformer, "blocks", None)
            if blocks is not None:
                for i, block in enumerate(blocks):
                    modules[f"{self.EVOFORMER_KEY}.blocks.{i}"] = block

        return modules

    def get_trunk_blocks(self) -> nn.ModuleList:
        """Evoformer blocks — the main repeating units of the trunk."""
        return self.model.evoformer.blocks

    def get_evoformer_blocks(self) -> nn.ModuleList:
        """Deprecated: use get_trunk_blocks() instead."""
        return self.get_trunk_blocks()

    def get_structure_module(self):
        return getattr(self.model, self.STRUCTURE_MODULE_KEY, None)

    def get_input_embedder(self):
        return getattr(self.model, self.INPUT_EMBEDDER_KEY, None)

    @property
    def default_peft_targets(self) -> list[str]:
        return ["linear_q", "linear_v"]

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        """
        OpenFold must remain in eval mode because both EvoformerStack._prep_blocks
        and TemplatePairStack.forward assert `not self.training` (chunked ops are
        inference-only). Gradients still flow through LoRA / unfrozen params via
        `requires_grad=True` and `torch.set_grad_enabled(True)`.
        """
        super().train(mode)
        self.model.eval()
        return self

    def freeze_trunk(self) -> None:
        """Freeze all model parameters (prep for PEFT or head-only training)."""
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_trunk(self) -> None:
        """Unfreeze all model parameters."""
        for p in self.model.parameters():
            p.requires_grad = True

    @property
    def peft_target_module(self) -> nn.Module:
        """Evoformer stack — where PEFT layers are injected."""
        return self.model.evoformer

    def freeze_except(self, *module_keys: str) -> None:
        """Freeze everything except the named modules."""
        self.freeze_trunk()
        targetable = self.get_targetable_modules()
        for key in module_keys:
            mod = targetable.get(key)
            if mod is None:
                raise KeyError(f"Unknown module key '{key}'. Available: {list(targetable)}")
            for p in mod.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    def param_summary(self) -> dict[str, int]:
        """Return total/trainable/frozen parameter counts."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def _remap_openfold_keys(state: dict, model: nn.Module) -> dict:
    """
    Remap checkpoint keys to match current OpenFold model structure.
    Handles naming changes between OpenFold versions (e.g. .core. prefix,
    IPA linear wrapping, template embedder nesting).
    """
    model_keys = set(model.state_dict().keys())
    fixed = {}

    for k, v in state.items():
        if k in model_keys:
            fixed[k] = v
            continue

        new_k = k
        new_k = re.sub(r"(blocks\.\d+)\.core\.(pair_transition|tri_mul_|tri_att_)",
                        r"\1.pair_stack.\2", new_k)
        new_k = re.sub(r"(blocks\.\d+)\.core\.", r"\1.", new_k)
        new_k = re.sub(r"(ipa\.linear_\w+_points)\.(weight|bias)",
                        r"\1.linear.\2", new_k)
        new_k = re.sub(r"^(template_(?:pair_embedder|pair_stack|angle_embedder|pointwise_att))",
                        r"template_embedder.\1", new_k)

        fixed[new_k] = v

    return fixed
