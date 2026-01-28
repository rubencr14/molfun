"""
src/benchmarks/bench_esm_fused_mlp12_synth_t30.py

Benchmark HuggingFace ESM-2 t30 (150M) baseline vs. patched MLP1+MLP2 using Triton.

Patched changes per encoder layer:
- MLP1: intermediate.forward(hidden_states) -> fused_linear_gelu_triton(hidden_states, W1, b1)
- MLP2: output.forward(hidden_states, input_tensor) -> fused_linear_bias_residual_triton(
          hidden_states, W2, b2, residual=input_tensor
        )
  ثم نُبقي LayerNorm / dropout كما في HF (dropout في eval() عادة no-op).

Usage:
  python src/benchmarks/bench_esm_fused_mlp12_synth_t30.py

Notes:
- Inference only.
- Uses CUDA events for accurate timing.
- Includes correctness checks (max/mean abs diff) on last_hidden_state.
- Uses synthetic "protein-like" sequences (canonical AA letters) for reproducibility.
- If you want to profile:
    nsys profile -o esm_patch_t30_mlp12 --force-overwrite true python .../bench_esm_fused_mlp12_synth_t30.py
"""

import types
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import torch
from transformers import AutoTokenizer, EsmModel

from src.kernels.fused_linear_gelu_triton import fused_linear_gelu_triton
from src.kernels.fused_linear_bias_residual_triton import fused_linear_bias_residual_triton


@dataclass
class BenchCfg:
    model_id: str = "facebook/esm2_t30_150M_UR50D"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    iters: int = 50
    warmup: int = 10


def time_it_cuda(fn, iters: int, warmup: int) -> float:
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
    return ms_total / iters  # ms/iter


def make_protein_like_sequences(batch_size: int, seq_len: int) -> List[str]:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"  # 20 canonical amino acids
    seq = (alphabet * ((seq_len // len(alphabet)) + 1))[:seq_len]
    return [seq for _ in range(batch_size)]


def patch_mlp12_fused(model: EsmModel):
    """
    Patch each encoder layer:
      - intermediate.forward(hidden_states) = fused_linear_gelu_triton(hidden_states, W1, b1)
      - output.forward(hidden_states, input_tensor) uses fused_linear_bias_residual_triton
        to fuse: dense2 + bias + residual-add. LayerNorm/dropout preserved.
    Returns list of (orig_intermediate_forward, orig_output_forward) so we can restore.
    """
    originals = []

    for layer in model.encoder.layer:
        intermediate = layer.intermediate
        output = layer.output

        originals.append((intermediate.forward, output.forward))

        # ---- Patch MLP1 (Dense1 + GELU) ----
        def new_intermediate_forward(self_intermediate, hidden_states):
            w1 = self_intermediate.dense.weight
            b1 = self_intermediate.dense.bias
            return fused_linear_gelu_triton(hidden_states, w1, b1)

        intermediate.forward = types.MethodType(new_intermediate_forward, intermediate)

        # ---- Patch MLP2 (Dense2 + bias + residual add) ----
        # HF ESM output forward is typically: forward(hidden_states, input_tensor)
        def new_output_forward(self_output, hidden_states, input_tensor):
            w2 = self_output.dense.weight
            b2 = self_output.dense.bias

            # Fuse: input_tensor + (hidden_states @ W2^T + b2)
            y = fused_linear_bias_residual_triton(hidden_states, w2, b2, residual=input_tensor)

            # Preserve HF semantics (dropout is usually no-op in eval())
            if hasattr(self_output, "dropout") and self_output.dropout is not None:
                y = self_output.dropout(y)

            # Preserve LayerNorm if present
            if hasattr(self_output, "LayerNorm") and self_output.LayerNorm is not None:
                y = self_output.LayerNorm(y)

            return y

        output.forward = types.MethodType(new_output_forward, output)

    return originals


def unpatch_mlp12(model: EsmModel, originals):
    """Restore original intermediate.forward and output.forward for each layer."""
    for layer, (orig_inter, orig_out) in zip(model.encoder.layer, originals):
        layer.intermediate.forward = orig_inter
        layer.output.forward = orig_out


@torch.inference_mode()
def forward_last_hidden(model: EsmModel, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(**batch).last_hidden_state


def run_benchmark() -> List[Dict[str, Any]]:
    cfg = BenchCfg()
    assert torch.cuda.is_available(), "CUDA required"
    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, do_lower_case=False)
    model = EsmModel.from_pretrained(cfg.model_id).to(cfg.device).eval().to(cfg.dtype)

    cases: List[Tuple[int, int]] = [
        (1, 256),
        (1, 512),
        (1, 1024),
        (2, 512),
        (2, 1024),
        (4, 512),
        (4, 1024),
        (8, 512),
    ]

    results = []

    for B, L in cases:
        seqs = make_protein_like_sequences(B, L)
        batch = tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        tokens = int(batch["attention_mask"].sum().item())

        # --- Baseline timing ---
        t_base = time_it_cuda(lambda: forward_last_hidden(model, batch), cfg.iters, cfg.warmup)

        # --- Patched timing (MLP1 + MLP2 fused) ---
        originals = patch_mlp12_fused(model)

        # Trigger Triton JIT/autotune once
        _ = forward_last_hidden(model, batch)
        torch.cuda.synchronize()

        t_pat = time_it_cuda(lambda: forward_last_hidden(model, batch), cfg.iters, cfg.warmup)

        # --- Correctness check ---
        unpatch_mlp12(model, originals)
        y_ref = forward_last_hidden(model, batch)

        originals = patch_mlp12_fused(model)
        y_pat = forward_last_hidden(model, batch)
        unpatch_mlp12(model, originals)

        max_abs = (y_ref - y_pat).abs().max().item()
        mean_abs = (y_ref - y_pat).abs().mean().item()

        speedup = t_base / t_pat if t_pat > 0 else float("inf")
        tok_per_s_base = tokens / (t_base / 1e3)
        tok_per_s_pat = tokens / (t_pat / 1e3)

        results.append({
            "benchmark_name": "esm_fused_mlp12_synth_t30",
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
                "tokens_per_sec_triton": round(tok_per_s_pat, 0),
                "model_id": cfg.model_id,
                "dtype": str(cfg.dtype),
                "device": cfg.device,
            }
        })

    return results


def main():
    cfg = BenchCfg()
    results = run_benchmark()

    print(f"Model: {cfg.model_id} | dtype: {cfg.dtype} | device: {cfg.device}")
    print(f"Iters: {cfg.iters}, warmup: {cfg.warmup}")
    print("Benchmark: Baseline vs Patched (fused MLP1 + fused MLP2+residual)\n")

    for result in results:
        meta = result["metadata"]
        print(f"B={meta['B']} L={meta['L']} | tokens={meta['tokens']}")
        print(f"  baseline: {result['baseline_time_ms']:.3f} ms/iter  ({meta['tokens_per_sec_baseline']:,.0f} tokens/s)")
        print(f"  patched : {result['triton_time_ms']:.3f} ms/iter  ({meta['tokens_per_sec_triton']:,.0f} tokens/s)")
        print(f"  speedup : {result['speedup']:.2f}x" if result['speedup'] else "  speedup : inf")
        print(f"  max|diff|: {result['max_diff']:.3e}  mean|diff|: {result['mean_diff']:.3e}\n")


if __name__ == "__main__":
    main()
