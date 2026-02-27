"""
TorchScript export for molfun models.

Exports models to TorchScript (traced or scripted) for deployment
without a Python runtime, on mobile, or with LibTorch C++.

Usage::

    from molfun.models import MolfunStructureModel
    from molfun.export import export_torchscript

    model = MolfunStructureModel("openfold", config=cfg, weights="ckpt.pt")
    model.merge()
    export_torchscript(model, "model.pt", seq_len=256)

    # Load and run without molfun installed:
    loaded = torch.jit.load("model.pt")
    out = loaded(aatype, residue_index)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Literal

import torch
import torch.nn as nn


class _ScriptWrapper(nn.Module):
    """Wraps adapter for TorchScript with a clean tensor-in tensor-out interface."""

    def __init__(self, adapter: nn.Module):
        super().__init__()
        self.adapter = adapter

    def forward(self, aatype: torch.Tensor, residue_index: torch.Tensor) -> torch.Tensor:
        batch = {"aatype": aatype, "residue_index": residue_index}
        out = self.adapter(batch)
        if hasattr(out, "single"):
            return out.single
        if isinstance(out, dict):
            for key in ("single", "s"):
                if key in out:
                    return out[key]
            return next(iter(out.values()))
        return out


def export_torchscript(
    model,
    path: str,
    seq_len: int = 256,
    mode: Literal["trace", "script"] = "trace",
    optimize: bool = True,
    device: str = "cpu",
    check: bool = True,
) -> Path:
    """
    Export a MolfunStructureModel to TorchScript.

    Args:
        model: MolfunStructureModel or nn.Module with adapter.
        path: Output .pt file path.
        seq_len: Sequence length for tracing dummy input.
        mode: "trace" (default, more compatible) or "script".
        optimize: Apply torch.jit.optimize_for_inference.
        device: Device for tracing ("cpu" recommended).
        check: Run a validation forward pass after export.

    Returns:
        Path to the exported TorchScript file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    adapter = _get_adapter(model)
    wrapper = _ScriptWrapper(adapter).to(device).eval()

    aatype = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    residue_index = torch.arange(seq_len, device=device).unsqueeze(0)

    with torch.no_grad():
        if mode == "trace":
            scripted = torch.jit.trace(wrapper, (aatype, residue_index))
        elif mode == "script":
            scripted = torch.jit.script(wrapper)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'trace' or 'script'.")

    if optimize:
        try:
            scripted = torch.jit.optimize_for_inference(scripted)
        except Exception:
            pass

    scripted.save(str(path))

    if check:
        loaded = torch.jit.load(str(path), map_location=device)
        with torch.no_grad():
            loaded(aatype, residue_index)

    return path


def _get_adapter(model) -> nn.Module:
    """Extract the adapter nn.Module from various model wrappers."""
    if hasattr(model, "adapter"):
        adapter = model.adapter
        if hasattr(adapter, "model"):
            return adapter.model
        return adapter
    if isinstance(model, nn.Module):
        return model
    raise TypeError(
        f"Cannot extract adapter from {type(model).__name__}. "
        "Pass a MolfunStructureModel or nn.Module."
    )
