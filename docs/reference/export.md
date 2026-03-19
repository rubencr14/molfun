# Export

Export Molfun models to ONNX and TorchScript formats for deployment in
production environments.

## Quick Start

```python
from molfun import MolfunStructureModel

model = MolfunStructureModel.from_pretrained("openfold_v2")

# Export to ONNX
model.export_onnx("model.onnx", opset_version=17)

# Export to TorchScript
model.export_torchscript("model.pt")

# Or use the standalone functions
from molfun.export import export_onnx, export_torchscript

export_onnx(model, "model.onnx")
export_torchscript(model, "model.pt")
```

## export_onnx

::: molfun.export.onnx.export_onnx
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Export a model to ONNX format.

```python
from molfun.export import export_onnx

export_onnx(
    model=model,
    output_path="model.onnx",
    opset_version=17,
    dynamic_axes={"sequence": {0: "batch", 1: "length"}},
    sample_sequence="MKFLILLFNILCLFPVLAADNH",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `MolfunStructureModel \| nn.Module` | *required* | Model to export |
| `output_path` | `str \| Path` | *required* | Output file path |
| `opset_version` | `int` | `17` | ONNX opset version |
| `dynamic_axes` | `dict \| None` | `None` | Dynamic axes for variable-length inputs |
| `sample_sequence` | `str \| None` | `None` | Sample sequence for tracing |
| `simplify` | `bool` | `True` | Run ONNX simplifier after export |

**Returns:** `Path` to the exported ONNX file.

### Loading an ONNX Model

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
inputs = {"aatype": aatype_np, "residue_index": residue_index_np}
outputs = session.run(None, inputs)
```

---

## export_torchscript

::: molfun.export.torchscript.export_torchscript
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Export a model to TorchScript format via tracing or scripting.

```python
from molfun.export import export_torchscript

export_torchscript(
    model=model,
    output_path="model.pt",
    method="trace",
    sample_sequence="MKFLILLFNILCLFPVLAADNH",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `MolfunStructureModel \| nn.Module` | *required* | Model to export |
| `output_path` | `str \| Path` | *required* | Output file path |
| `method` | `str` | `"trace"` | Export method: `"trace"` or `"script"` |
| `sample_sequence` | `str \| None` | `None` | Sample sequence for tracing |
| `optimize` | `bool` | `True` | Apply TorchScript optimizations |

**Returns:** `Path` to the exported TorchScript file.

### Loading a TorchScript Model

```python
import torch

model = torch.jit.load("model.pt")
output = model(aatype, residue_index)
```

---

## Comparison

| Feature | ONNX | TorchScript |
|---------|------|-------------|
| Runtime | ONNX Runtime, TensorRT | PyTorch, LibTorch |
| Language support | Python, C++, C#, Java, JS | Python, C++ |
| Optimization | Graph optimizations, quantization | JIT optimizations |
| Dynamic shapes | Via dynamic_axes | Native support |
| Best for | Cross-platform deployment | PyTorch ecosystem |

## CLI Export

```bash
# ONNX
molfun run export --format onnx --model openfold_v2 --output model.onnx

# TorchScript
molfun run export --format torchscript --model openfold_v2 --output model.pt
```
