"""
LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

Uses HuggingFace PEFT as backend for battle-tested LoRA, IA³, etc.
Falls back to a lightweight built-in implementation when PEFT is not installed.
"""

from __future__ import annotations
import math
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from peft import LoraConfig, IA3Config, get_peft_model, PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


# =========================================================================
# Built-in LoRA (zero dependencies)
# =========================================================================

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with low-rank adaptation.
    W_effective = W_frozen + (alpha/rank) * A @ B
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = original.weight
        self.weight.requires_grad = False
        self.bias = original.bias
        if self.bias is not None:
            self.bias.requires_grad = False

        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, device=self.weight.device, dtype=self.weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, device=self.weight.device, dtype=self.weight.dtype) #initialize the LoRA B matrix to 0
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) #randomly initialize the LoRA matrix
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._merged:
            return F.linear(x, self.weight, self.bias)
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base + lora * self.scaling

    def merge(self) -> None:
        if not self._merged:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self._merged = True

    def unmerge(self) -> None:
        if self._merged:
            self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self._merged = False


# =========================================================================
# Unified PEFT wrapper
# =========================================================================

class MolfunPEFT:
    """
    Unified PEFT interface. Uses HuggingFace PEFT when available,
    falls back to built-in LoRALinear otherwise.
    
    Supports: LoRA, IA³ (via HF PEFT), built-in LoRA (fallback).
    
    Usage:
        adapter = OpenFoldAdapter(model=model)
        
        # LoRA
        peft = MolfunPEFT.lora(rank=8, target_modules=["linear_q", "linear_v"])
        peft.apply(adapter.model.evoformer)
        
        # IA³ (requires HF PEFT)
        peft = MolfunPEFT.ia3(target_modules=["linear_v"], feedforward_modules=["ff_linear1"])
        peft.apply(adapter.model.evoformer)
        
        # Training: only adapted params
        optimizer = torch.optim.Adam(peft.trainable_parameters(), lr=1e-4)
        
        # Export: merge into base weights
        peft.merge()
    """

    def __init__(
        self,
        method: str,
        config: dict,
        use_hf: bool = True,
    ):
        self.method = method
        self.config = config
        self.use_hf = use_hf and HAS_PEFT
        self._model: Optional[nn.Module] = None
        self._peft_model: Optional[PeftModel] = None
        self._builtin_layers: list[tuple[nn.Module, str, LoRALinear]] = []

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def lora(
        cls,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
        use_hf: bool = True,
    ) -> MolfunPEFT:
        """Create a LoRA adapter."""
        target_modules = target_modules or ["linear_q", "linear_k", "linear_v"]
        return cls(
            method="lora",
            config={
                "rank": rank,
                "alpha": alpha,
                "dropout": dropout,
                "target_modules": target_modules,
            },
            use_hf=use_hf,
        )

    @classmethod
    def ia3(
        cls,
        target_modules: Optional[list[str]] = None,
        feedforward_modules: Optional[list[str]] = None,
    ) -> MolfunPEFT:
        """Create an IA³ adapter (requires HuggingFace PEFT)."""
        if not HAS_PEFT:
            raise ImportError("IA³ requires HuggingFace PEFT: pip install peft")
        target_modules = target_modules or ["linear_q", "linear_v"]
        feedforward_modules = feedforward_modules or []
        return cls(
            method="ia3",
            config={
                "target_modules": target_modules,
                "feedforward_modules": feedforward_modules,
            },
            use_hf=True,
        )

    # ------------------------------------------------------------------
    # Apply to model
    # ------------------------------------------------------------------

    def apply(self, model: nn.Module) -> nn.Module:
        """
        Apply PEFT method to the model. Freezes base params automatically.
        
        Args:
            model: nn.Module to adapt (e.g. adapter.model or adapter.model.evoformer).
            
        Returns:
            The adapted model (may be wrapped by PeftModel if using HF backend).
        """
        self._model = model

        if self.use_hf and self.method == "lora":
            return self._apply_hf_lora(model)
        elif self.use_hf and self.method == "ia3":
            return self._apply_hf_ia3(model)
        else:
            return self._apply_builtin_lora(model)

    def _apply_hf_lora(self, model: nn.Module) -> nn.Module:
        lora_config = LoraConfig(
            r=self.config["rank"],
            lora_alpha=self.config["alpha"],
            lora_dropout=self.config["dropout"],
            target_modules=self.config["target_modules"],
            bias="none",
        )
        self._peft_model = get_peft_model(model, lora_config)
        return self._peft_model

    def _apply_hf_ia3(self, model: nn.Module) -> nn.Module:
        ia3_config = IA3Config(
            target_modules=self.config["target_modules"],
            feedforward_modules=self.config.get("feedforward_modules", []),
        )
        self._peft_model = get_peft_model(model, ia3_config)
        return self._peft_model

    def _apply_builtin_lora(self, model: nn.Module) -> nn.Module:
        """Fallback: inject LoRALinear layers manually if PEFT is not installed."""
        for p in model.parameters():
            p.requires_grad = False

        targets = self.config["target_modules"]
        for parent_name, parent_module in model.named_modules():
            for attr_name, child in list(parent_module._modules.items()):
                if not isinstance(child, nn.Linear):
                    continue
                full_name = f"{parent_name}.{attr_name}" if parent_name else attr_name
                if not any(t in full_name for t in targets):
                    continue
                lora_layer = LoRALinear(
                    child,
                    rank=self.config["rank"],
                    alpha=self.config["alpha"],
                    dropout=self.config["dropout"],
                )
                parent_module._modules[attr_name] = lora_layer
                self._builtin_layers.append((parent_module, full_name, lora_layer))

        return model

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable (PEFT) parameters."""
        if self._peft_model is not None:
            return [p for p in self._peft_model.parameters() if p.requires_grad]
        return [p for _, _, layer in self._builtin_layers for p in [layer.lora_A, layer.lora_B]]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    # ------------------------------------------------------------------
    # Merge / unmerge
    # ------------------------------------------------------------------

    def merge(self) -> None:
        """Merge adapted weights into base model (for inference/export)."""
        if self._peft_model is not None:
            self._peft_model.merge_adapter()
        else:
            for _, _, layer in self._builtin_layers:
                layer.merge()

    def unmerge(self) -> None:
        """Restore base weights (undo merge)."""
        if self._peft_model is not None:
            self._peft_model.unmerge_adapter()
        else:
            for _, _, layer in self._builtin_layers:
                layer.unmerge()

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save PEFT adapter weights."""
        if self._peft_model is not None:
            self._peft_model.save_pretrained(path)
        else:
            state = {}
            for _, full_name, layer in self._builtin_layers:
                state[f"{full_name}.lora_A"] = layer.lora_A.data.cpu()
                state[f"{full_name}.lora_B"] = layer.lora_B.data.cpu()
            torch.save(state, path)

    def load(self, path: str) -> None:
        """Load PEFT adapter weights."""
        if self._peft_model is not None:
            from peft import PeftModel as PM
            self._peft_model.load_adapter(path, adapter_name="default")
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
            for _, full_name, layer in self._builtin_layers:
                layer.lora_A.data = state[f"{full_name}.lora_A"].to(layer.lora_A.device)
                layer.lora_B.data = state[f"{full_name}.lora_B"].to(layer.lora_B.device)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        return "huggingface_peft" if (self._peft_model is not None) else "builtin"

    def summary(self) -> dict:
        total = sum(p.numel() for p in (self._peft_model or self._model).parameters()) if (self._peft_model or self._model) else 0
        trainable = self.trainable_param_count()
        return {
            "method": self.method,
            "backend": self.backend,
            "config": self.config,
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": (trainable / total * 100) if total > 0 else 0,
        }
