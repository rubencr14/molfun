"""
ONNX export for molfun models.

Exports structure prediction models to ONNX format for
optimized inference with ONNX Runtime, NVIDIA TensorRT, etc.

Usage::

    from molfun.models import MolfunStructureModel
    from molfun.export import export_onnx

    model = MolfunStructureModel("openfold", config=cfg, weights="ckpt.pt")
    model.merge()  # merge LoRA weights first if applicable
    export_onnx(model, "model.onnx", seq_len=256)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class _TracingWrapper(nn.Module):
    """Wraps an adapter for ONNX tracing with a fixed-shape input contract."""

    def __init__(self, adapter: nn.Module, output_keys: list[str]):
        super().__init__()
        self.adapter = adapter
        self.output_keys = output_keys

    def forward(self, aatype: torch.Tensor, residue_index: torch.Tensor):
        batch = {"aatype": aatype, "residue_index": residue_index}
        out = self.adapter(batch)
        if hasattr(out, "single"):
            return out.single
        if isinstance(out, dict):
            return out.get("single", out.get("s", next(iter(out.values()))))
        return out


def export_onnx(
    model,
    path: str,
    seq_len: int = 256,
    opset_version: int = 17,
    dynamic_axes: Optional[dict] = None,
    simplify: bool = False,
    check: bool = True,
    device: str = "cpu",
) -> Path:
    """
    Export a MolfunStructureModel to ONNX format.

    Args:
        model: MolfunStructureModel or nn.Module with adapter.
        path: Output .onnx file path.
        seq_len: Sequence length for the dummy input.
        opset_version: ONNX opset version (default 17).
        dynamic_axes: Custom dynamic axes mapping. If None, seq_len
                      dimension is dynamic by default.
        simplify: Run onnx-simplifier after export (requires onnxsim).
        check: Validate the exported model with onnx.checker.
        device: Device for tracing ("cpu" recommended).

    Returns:
        Path to the exported ONNX file.

    Raises:
        ImportError: If onnx package is not installed.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError(
            "ONNX export requires the 'onnx' package. "
            "Install with: pip install onnx"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    adapter = _get_adapter(model)
    wrapper = _TracingWrapper(adapter, output_keys=["single"])
    wrapper = wrapper.to(device).eval()

    aatype = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    residue_index = torch.arange(seq_len, device=device).unsqueeze(0)

    if dynamic_axes is None:
        dynamic_axes = {
            "aatype": {0: "batch", 1: "seq_len"},
            "residue_index": {0: "batch", 1: "seq_len"},
            "output": {0: "batch", 1: "seq_len"},
        }

    torch.onnx.export(
        wrapper,
        (aatype, residue_index),
        str(path),
        opset_version=opset_version,
        input_names=["aatype", "residue_index"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    if check:
        onnx_model = onnx.load(str(path))
        onnx.checker.check_model(onnx_model)

    if simplify:
        try:
            import onnxsim
            onnx_model = onnx.load(str(path))
            simplified, ok = onnxsim.simplify(onnx_model)
            if ok:
                onnx.save(simplified, str(path))
        except ImportError:
            pass

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
