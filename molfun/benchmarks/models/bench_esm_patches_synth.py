"""
molfun/benchmarks/bench_esm_patches_synth.py

One benchmark script to compare:
- baseline (stock HF ESM forward)
- mlp1      : fused_linear_gelu_triton on intermediate (Dense1 + Bias + GELU)
- mlp12     : mlp1 + fused_linear_bias_residual_triton on output (Dense2 + Bias + Residual Add)
- ln        : Triton LayerNorm for all nn.LayerNorm modules inside encoder layers
- mlp12_ln  : mlp12 + ln combined

Synthetic sequences (canonical AA letters) => reproducible.

Usage:
  python molfun/benchmarks/bench_esm_patches_synth.py
  python molfun/benchmarks/bench_esm_patches_synth.py --model_id facebook/esm2_t30_150M_UR50D --mode mlp12_ln
  python molfun/benchmarks/bench_esm_patches_synth.py --cases "1,256;1,512;2,1024"

Notes:
- Inference only.
- CUDA events for timing.
- Correctness check compares last_hidden_state baseline vs patched.

Required kernels in repo:
- molfun/kernels/models/fused_linear_gelu_triton.py
- molfun/kernels/models/fused_linear_bias_residual_triton.py
- molfun/kernels/models/layernorm_triton.py  (must expose layernorm_triton(x, gamma, beta, eps))
"""

import argparse
import types
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Callable

import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel

from molfun.kernels.models.fused_linear_gelu_triton import fused_linear_gelu_triton
from molfun.kernels.models.fused_linear_bias_residual_triton import fused_linear_bias_residual_triton
from molfun.kernels.models.layernorm_triton import layernorm_triton



# -------------------------- Config --------------------------

@dataclass
class BenchCfg:
    model_id: str = "facebook/esm2_t30_150M_UR50D"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    iters: int = 50
    warmup: int = 10


# -------------------------- Timing --------------------------

def time_it_cuda(fn: Callable[[], torch.Tensor], iters: int, warmup: int) -> float:
    # Warmup (important: triggers Triton JIT/autotune in patched mode)
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = fn()
    end.record()
    torch.cuda.synchronize()

    ms_total = start.elapsed_time(end)
    return ms_total / iters


# -------------------------- Synthetic inputs --------------------------

def make_protein_like_sequences(batch_size: int, seq_len: int) -> List[str]:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"  # 20 canonical amino acids
    seq = (alphabet * ((seq_len // len(alphabet)) + 1))[:seq_len]
    return [seq for _ in range(batch_size)]


@torch.inference_mode()
def forward_last_hidden(model: EsmModel, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(**batch).last_hidden_state


# -------------------------- Patches --------------------------

def patch_mlp1_fused(model: EsmModel):
    """
    Patch each encoder layer so that:
      intermediate.forward(hidden_states) == fused_linear_gelu_triton(hidden_states, W1, b1)
    Returns list of original intermediate.forward callables.
    """
    originals = []

    for layer in model.encoder.layer:
        intermediate = layer.intermediate
        originals.append(intermediate.forward)

        def make_new_forward(intermediate_module):
            def new_forward(self_intermediate, hidden_states):
                w = intermediate_module.dense.weight
                b = intermediate_module.dense.bias
                return fused_linear_gelu_triton(hidden_states, w, b)
            return new_forward

        intermediate.forward = types.MethodType(make_new_forward(intermediate), intermediate)

    return originals


def unpatch_mlp1(model: EsmModel, originals):
    for layer, orig in zip(model.encoder.layer, originals):
        layer.intermediate.forward = orig


def patch_mlp12_fused(model: EsmModel):
    """
    Patch each encoder layer:
      - intermediate.forward(hidden_states) = fused_linear_gelu_triton(hidden_states, W1, b1)
      - output.forward(hidden_states, input_tensor) = LayerNorm(dropout(residual + dense2(hidden_states)))
        where residual + dense2 is fused into one Triton kernel.
    Returns list of (orig_intermediate_forward, orig_output_forward).
    """
    originals = []

    for layer in model.encoder.layer:
        intermediate = layer.intermediate
        output = layer.output
        originals.append((intermediate.forward, output.forward))

        # ---- MLP1 ----
        def make_inter_forward(intermediate_module):
            def new_intermediate_forward(self_intermediate, hidden_states):
                w1 = intermediate_module.dense.weight
                b1 = intermediate_module.dense.bias
                return fused_linear_gelu_triton(hidden_states, w1, b1)
            return new_intermediate_forward

        intermediate.forward = types.MethodType(make_inter_forward(intermediate), intermediate)

        # ---- MLP2 + residual add ----
        def make_out_forward(output_module):
            def new_output_forward(self_output, hidden_states, input_tensor):
                w2 = output_module.dense.weight
                b2 = output_module.dense.bias

                y = fused_linear_bias_residual_triton(hidden_states, w2, b2, residual=input_tensor)

                # preserve HF semantics
                if hasattr(output_module, "dropout") and output_module.dropout is not None:
                    y = output_module.dropout(y)  # eval() => typically no-op
                if hasattr(output_module, "LayerNorm") and output_module.LayerNorm is not None:
                    y = output_module.LayerNorm(y)
                return y
            return new_output_forward

        output.forward = types.MethodType(make_out_forward(output), output)

    return originals


def unpatch_mlp12(model: EsmModel, originals):
    for layer, (orig_inter, orig_out) in zip(model.encoder.layer, originals):
        layer.intermediate.forward = orig_inter
        layer.output.forward = orig_out


def patch_layernorm_triton_encoder(model: EsmModel):
    """
    Patch ALL nn.LayerNorm modules inside encoder layers to call layernorm_triton.
    Returns list of (module, original_forward).
    """
    originals: List[Tuple[nn.LayerNorm, Callable]] = []

    for layer in model.encoder.layer:
        for mod in layer.modules():
            if isinstance(mod, nn.LayerNorm):
                originals.append((mod, mod.forward))

                def make_ln_forward(ln_module: nn.LayerNorm):
                    def new_ln_forward(self_ln, x):
                        return layernorm_triton(x, ln_module.weight, ln_module.bias, eps=float(ln_module.eps))
                    return new_ln_forward

                mod.forward = types.MethodType(make_ln_forward(mod), mod)

    return originals


def unpatch_layernorm(originals):
    for mod, orig_fwd in originals:
        mod.forward = orig_fwd


# -------------------------- Runner --------------------------

def apply_patch_mode(model: EsmModel, mode: str):
    """
    Apply selected patch mode. Returns a callable "restore()" that unpatches everything applied.
    Modes: baseline, mlp1, mlp12, ln, mlp12_ln
    """
    restores: List[Callable[[], None]] = []

    if mode == "baseline":
        return lambda: None

    if mode == "mlp1":
        orig = patch_mlp1_fused(model)
        restores.append(lambda: unpatch_mlp1(model, orig))
        return lambda: [r() for r in restores]

    if mode == "mlp12":
        orig = patch_mlp12_fused(model)
        restores.append(lambda: unpatch_mlp12(model, orig))
        return lambda: [r() for r in restores]

    if mode == "ln":
        orig = patch_layernorm_triton_encoder(model)
        restores.append(lambda: unpatch_layernorm(orig))
        return lambda: [r() for r in restores]

    if mode == "mlp12_ln":
        orig_mlp = patch_mlp12_fused(model)
        restores.append(lambda: unpatch_mlp12(model, orig_mlp))

        orig_ln = patch_layernorm_triton_encoder(model)
        restores.append(lambda: unpatch_layernorm(orig_ln))

        return lambda: [r() for r in restores]

    raise ValueError(f"Unknown mode: {mode}")


def parse_cases(s: str) -> List[Tuple[int, int]]:
    """
    Format: "1,256;1,512;2,1024"
    """
    out: List[Tuple[int, int]] = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        b_str, l_str = part.split(",")
        out.append((int(b_str), int(l_str)))
    return out


def run_benchmark(cfg: BenchCfg, cases: List[Tuple[int, int]], mode: str) -> List[Dict[str, Any]]:
    assert torch.cuda.is_available(), "CUDA required"
    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, do_lower_case=False)
    model = EsmModel.from_pretrained(cfg.model_id).to(cfg.device).eval().to(cfg.dtype)

    results: List[Dict[str, Any]] = []

    for B, L in cases:
        seqs = make_protein_like_sequences(B, L)
        batch = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        tokens = int(batch["attention_mask"].sum().item())

        # --- baseline timing ---
        t_base = time_it_cuda(lambda: forward_last_hidden(model, batch), cfg.iters, cfg.warmup)

        # --- patched timing ---
        restore = apply_patch_mode(model, mode)

        # trigger JIT/autotune once
        _ = forward_last_hidden(model, batch)
        torch.cuda.synchronize()

        t_pat = time_it_cuda(lambda: forward_last_hidden(model, batch), cfg.iters, cfg.warmup)

        # --- correctness ---
        restore()
        y_ref = forward_last_hidden(model, batch)

        restore2 = apply_patch_mode(model, mode)
        y_pat = forward_last_hidden(model, batch)
        restore2()

        max_abs = (y_ref - y_pat).abs().max().item()
        mean_abs = (y_ref - y_pat).abs().mean().item()

        speedup = t_base / t_pat if t_pat > 0 else float("inf")
        tok_per_s_base = tokens / (t_base / 1e3)
        tok_per_s_pat = tokens / (t_pat / 1e3)

        results.append({
            "benchmark_name": "esm_patches_synth",
            "case_name": f"B={B}_L={L}",
            "baseline_time_ms": round(t_base, 3),
            "triton_time_ms": round(t_pat, 3),
            "speedup": round(speedup, 2) if speedup != float("inf") else None,
            "max_diff": max_abs,
            "mean_diff": mean_abs,
            "metadata": {
                "B": B,
                "L": L,
                "tokens": tokens,
                "tokens_per_sec_baseline": round(tok_per_s_base, 0),
                "tokens_per_sec_patched": round(tok_per_s_pat, 0),
                "model_id": cfg.model_id,
                "dtype": str(cfg.dtype),
                "device": cfg.device,
                "mode": mode,
            }
        })

    return results


# Wrapper function for API compatibility (no arguments)
def run_benchmark_api() -> List[Dict[str, Any]]:
    """Wrapper function for API compatibility - uses default values"""
    cfg = BenchCfg()
    cases = parse_cases("1,256;1,512;1,1024;2,512;2,1024;4,512;4,1024;8,512")
    mode = "mlp12_ln"  # Default mode
    return run_benchmark(cfg, cases, mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--mode", type=str, default="mlp12_ln",
                        choices=["baseline", "mlp1", "mlp12", "ln", "mlp12_ln"])
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cases", type=str, default="1,256;1,512;1,1024;2,512;2,1024;4,512;4,1024;8,512")
    args = parser.parse_args()

    cfg = BenchCfg(
        model_id=args.model_id,
        device="cuda",
        dtype=torch.float16 if args.dtype == "fp16" else torch.bfloat16,
        iters=args.iters,
        warmup=args.warmup,
    )
    cases = parse_cases(args.cases)

    results = run_benchmark(cfg, cases, args.mode)

    print(f"Model: {cfg.model_id} | dtype: {cfg.dtype} | device: {cfg.device}")
    print(f"Iters: {cfg.iters}, warmup: {cfg.warmup}")
    print(f"Benchmark: Baseline vs Patched (mode={args.mode})\n")

    for r in results:
        meta = r["metadata"]
        print(f"B={meta['B']} L={meta['L']} | tokens={meta['tokens']}")
        print(f"  baseline: {r['baseline_time_ms']:.3f} ms/iter  ({meta['tokens_per_sec_baseline']:,.0f} tokens/s)")
        print(f"  patched : {r['triton_time_ms']:.3f} ms/iter  ({meta['tokens_per_sec_patched']:,.0f} tokens/s)")
        print(f"  speedup : {r['speedup']:.2f}x" if r["speedup"] else "  speedup : inf")
        print(f"  max|diff|: {r['max_diff']:.3e}  mean|diff|: {r['mean_diff']:.3e}\n")


if __name__ == "__main__":
    main()
