"""
molfun/benchmarks/bench_esm_fused_mlp1_synth.py

Benchmark HuggingFace ESM baseline vs. patched MLP1 using Triton fused Linear+Bias+GELU.
This version uses ONLY synthetic protein-like sequences (valid amino-acid letters),
so it is fully reproducible and does not depend on FASTA files.

Usage:
  python molfun/benchmarks/bench_esm_fused_mlp1_synth.py

What it measures:
- Baseline: stock EsmModel forward pass
- Patched : same model, but the first MLP projection per layer (intermediate.dense + GELU)
            is replaced with fused_linear_gelu_triton(hidden_states, W, b)

Notes:
- Inference only.
- Uses CUDA events for accurate timing.
- Includes correctness checks (max/mean abs diff) on last_hidden_state.
- If you want to profile, run:
    nsys profile -o esm_patch --force-overwrite true python .../bench_esm_fused_mlp1_synth.py
"""

import types
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import torch
from transformers import AutoTokenizer, EsmModel

from molfun.kernels.models.fused_linear_gelu_triton import fused_linear_gelu_triton


@dataclass
class BenchCfg:
    model_id: str = "facebook/esm2_t6_8M_UR50D"  # small, good for iteration
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    iters: int = 50
    warmup: int = 10


def time_it_cuda(fn, iters: int, warmup: int) -> float:
    # Warmup (important: triggers Triton JIT/autotune in patched mode)
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    # CUDA events give accurate GPU timing (better than time.perf_counter for micro timings)
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
    """
    Create deterministic synthetic sequences using canonical amino acid letters.
    These are "protein-like" in the sense they use valid AA tokens, but they are not real proteins.
    For performance benchmarking this is sufficient because runtime depends on tensor shapes (B, L),
    not on biological realism.
    """
    alphabet = "ACDEFGHIKLMNPQRSTVWY"  # 20 canonical amino acids
    seq = (alphabet * ((seq_len // len(alphabet)) + 1))[:seq_len]
    return [seq for _ in range(batch_size)]


def patch_mlp1_fused(model: EsmModel):
    """
    Patch each encoder layer so that:
        intermediate.forward(hidden_states) == fused_linear_gelu_triton(hidden_states, W, b)

    In HF ESM, layer.intermediate.dense is the first MLP Linear (D -> 4D), and the module's forward
    applies dense + activation. By replacing intermediate.forward, we fuse dense+bias+gelu in one kernel.

    Returns a list of original forward callables so we can restore them after benchmarking.
    """
    originals = []

    for layer in model.encoder.layer:
        intermediate = layer.intermediate

        # Save the original forward so we can restore
        originals.append(intermediate.forward)

        # Define patched forward
        def new_forward(self_intermediate, hidden_states):
            # hidden_states: [B, T, D]
            # dense.weight layout: [4D, D] (out_features, in_features)
            # dense.bias layout:   [4D]
            w = self_intermediate.dense.weight
            b = self_intermediate.dense.bias
            return fused_linear_gelu_triton(hidden_states, w, b)

        # Bind the function to this module instance (so "self_intermediate" works)
        intermediate.forward = types.MethodType(new_forward, intermediate)

    return originals


def unpatch_mlp1(model: EsmModel, originals):
    """
    Restore original intermediate.forward methods for each layer.
    """
    for layer, orig in zip(model.encoder.layer, originals):
        layer.intermediate.forward = orig


@torch.inference_mode()
def forward_last_hidden(model: EsmModel, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Run a forward pass and return last_hidden_state.
    """
    return model(**batch).last_hidden_state


def run_benchmark() -> List[Dict[str, Any]]:
    """Ejecuta el benchmark y devuelve resultados estructurados"""
    cfg = BenchCfg()
    assert torch.cuda.is_available(), "CUDA required"
    torch.set_grad_enabled(False)

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, do_lower_case=False)
    model = EsmModel.from_pretrained(cfg.model_id).to(cfg.device).eval().to(cfg.dtype)

    # Benchmark cases: (batch_size, sequence_length)
    # Keep these realistic for ESM-2 small; increase as desired.
    cases: List[Tuple[int, int]] = [
        (1, 256),
        (1, 512),
        (1, 1024),
        (1, 2048),
        (2, 1024),
        (4, 1024),
        (8, 1024),
        (16, 1024),
    ]

    results = []

    for B, L in cases:
        # Create synthetic protein-like sequences
        seqs = make_protein_like_sequences(B, L)

        # Tokenize (padding is trivial here since all sequences are same length)
        batch = tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        batch = {k: v.to(cfg.device) for k, v in batch.items()}

        # Approx token count (includes special tokens, but that's fine for throughput comparisons)
        tokens = int(batch["attention_mask"].sum().item())

        # --- Baseline timing (stock model) ---
        t_base = time_it_cuda(lambda: forward_last_hidden(model, batch), cfg.iters, cfg.warmup)

        # --- Patched timing (fused MLP1) ---
        originals = patch_mlp1_fused(model)

        # Trigger Triton JIT/autotune once before timing
        _ = forward_last_hidden(model, batch)
        torch.cuda.synchronize()

        t_pat = time_it_cuda(lambda: forward_last_hidden(model, batch), cfg.iters, cfg.warmup)

        # --- Correctness check ---
        # Restore baseline and compute reference output
        unpatch_mlp1(model, originals)
        y_ref = forward_last_hidden(model, batch)

        # Re-patch and compute patched output
        originals = patch_mlp1_fused(model)
        y_pat = forward_last_hidden(model, batch)
        unpatch_mlp1(model, originals)

        max_abs = (y_ref - y_pat).abs().max().item()
        mean_abs = (y_ref - y_pat).abs().mean().item()

        speedup = t_base / t_pat if t_pat > 0 else float("inf")
        tok_per_s_base = tokens / (t_base / 1e3)  # tokens/sec
        tok_per_s_pat = tokens / (t_pat / 1e3)

        results.append({
            "benchmark_name": "esm_fused_mlp1_synth",
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
    results = run_benchmark()
    cfg = BenchCfg()
    
    print(f"Model: {cfg.model_id} | dtype: {cfg.dtype} | device: {cfg.device}")
    print(f"Iters: {cfg.iters}, warmup: {cfg.warmup}")
    print("Benchmark: Baseline vs Patched (fused MLP1)\n")

    for result in results:
        meta = result["metadata"]
        print(f"B={meta['B']} L={meta['L']} | tokens={meta['tokens']}")
        print(f"  baseline: {result['baseline_time_ms']:.3f} ms/iter  ({meta['tokens_per_sec_baseline']:,.0f} tokens/s)")
        print(f"  patched : {result['triton_time_ms']:.3f} ms/iter  ({meta['tokens_per_sec_triton']:,.0f} tokens/s)")
        print(f"  speedup : {result['speedup']:.2f}x" if result['speedup'] else "  speedup : inf")
        print(f"  max|diff|: {result['max_diff']:.3e}  mean|diff|: {result['mean_diff']:.3e}\n")


if __name__ == "__main__":
    main()
