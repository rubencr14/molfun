"""
molfun/benchmarks/bench_esmfold_synth.py

Benchmark ESMFold baseline vs. patched MLP1 (stem) using Triton fused Linear+Bias+GELU.
This version uses ONLY synthetic protein-like sequences (valid amino-acid letters),
so it is fully reproducible and does not depend on FASTA files.

Usage:
  python molfun/benchmarks/bench_esmfold_synth.py

What it measures:
- Baseline: stock EsmForProteinFolding forward pass (num_recycles=0 for latency)
- Patched : same model, but the ESM-2 stem MLP1 (intermediate.dense + GELU) is replaced
            with fused_linear_gelu_triton(hidden_states, W, b)

Notes:
- Inference only.
- Uses CUDA events for accurate timing.
- Includes correctness checks (max/mean abs diff) on positions output.
- num_recycles=0 is used for latency benchmarking (no recycling loop).
- If you want to profile, run:
    nsys profile -o esmfold_patch --force-overwrite true python .../bench_esmfold_synth.py
"""

import types
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

from molfun.kernels.fused_linear_gelu_triton import fused_linear_gelu_triton


@dataclass
class BenchCfg:
    model_id: str = "facebook/esmfold_v1"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    iters: int = 50
    warmup: int = 10
    num_recycles: int = 0  # 0 for latency benchmarking


def time_it_cuda(fn, iters: int, warmup: int) -> float:
    # Warmup (important: triggers Triton JIT/autotune in patched mode)
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    # CUDA events give accurate GPU timing
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
    """
    alphabet = "ACDEFGHIKLMNPQRSTVWY"  # 20 canonical amino acids
    seq = (alphabet * ((seq_len // len(alphabet)) + 1))[:seq_len]
    return [seq for _ in range(batch_size)]


def patch_esmfold_stem_mlp1(model: EsmForProteinFolding):
    """
    Patch the ESM-2 stem MLP1 layers in ESMFold.
    
    The stem is at model.esm, and we patch intermediate.forward for each encoder layer
    to use fused_linear_gelu_triton instead of the default implementation.
    
    Returns a list of original forward callables so we can restore them after benchmarking.
    """
    originals = []
    stem = getattr(model, "esm", None)
    
    if stem is None:
        raise ValueError("Could not find ESM-2 stem at model.esm")
    
    for layer in stem.encoder.layer:
        intermediate = layer.intermediate
        
        # Save the original forward so we can restore
        originals.append(intermediate.forward)
        
        # Define patched forward
        def make_new_forward(intermediate_module):
            def new_forward(self_intermediate, hidden_states):
                w = intermediate_module.dense.weight
                b = intermediate_module.dense.bias
                return fused_linear_gelu_triton(hidden_states, w, b)
            return new_forward
        
        # Bind the function to this module instance
        intermediate.forward = types.MethodType(make_new_forward(intermediate), intermediate)
    
    return originals


def unpatch_esmfold_stem_mlp1(model: EsmForProteinFolding, originals):
    """
    Restore original intermediate.forward methods for each layer in the ESM stem.
    """
    stem = model.esm
    for layer, orig in zip(stem.encoder.layer, originals):
        layer.intermediate.forward = orig


@torch.inference_mode()
def forward_esmfold(model: EsmForProteinFolding, batch: Dict[str, torch.Tensor], num_recycles: int = 0):
    """
    Run ESMFold forward pass and return positions output.
    
    ESMFold expects:
    - input_ids: token IDs
    - attention_mask: attention mask (optional but recommended)
    - position_ids: position IDs (optional, will be created if not provided)
    - num_recycles: number of recycling iterations (0 for latency benchmarking)
    """
    # Only pass the keys that ESMFold expects
    # Remove position_ids if it causes issues, ESMFold can generate it internally
    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch.get("attention_mask"),
    }
    # Only add position_ids if it exists and is valid
    if "position_ids" in batch:
        model_inputs["position_ids"] = batch["position_ids"]
    
    out = model(**model_inputs, num_recycles=num_recycles)
    return out.positions  # Force use of result


def run_benchmark() -> List[Dict[str, Any]]:
    """Ejecuta el benchmark y devuelve resultados estructurados"""
    cfg = BenchCfg()
    assert torch.cuda.is_available(), "CUDA required"
    torch.set_grad_enabled(False)

    # Load tokenizer + model
    # ESMFold uses a specific tokenizer that handles amino acid sequences
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, do_lower_case=False)
    model = EsmForProteinFolding.from_pretrained(
        cfg.model_id,
        use_safetensors=False,
    ).to(cfg.device).eval().to(cfg.dtype)
    
    # Verify tokenizer vocab size matches model
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model vocab size: {model.config.vocab_size}")

    # Benchmark cases: (batch_size, sequence_length)
    # ESMFold can handle longer sequences, adjust as needed
    cases: List[Tuple[int, int]] = [
        (1, 128),
        (1, 256),
        (1, 512),
        (2, 256),
        (2, 512),
        (4, 256),
    ]

    results = []

    for B, L in cases:
        # Create synthetic protein-like sequences
        seqs = make_protein_like_sequences(B, L)

        # Tokenize - ESMFold tokenizer handles amino acid sequences
        # Note: ESMFold can handle sequences up to ~1024 amino acids
        # We use truncation to ensure we don't exceed model limits
        batch = tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,  # ESMFold max length
            add_special_tokens=True,  # Explicitly add BOS/EOS if needed
        )
        batch = {k: v.to(cfg.device) for k, v in batch.items()}

        # Validate input_ids are within vocab range
        input_ids = batch["input_ids"]
        vocab_size = model.config.vocab_size
        min_id = input_ids.min().item()
        max_id = input_ids.max().item()
        
        # Debug: print IDs info (can be removed later)
        print(f"Debug: B={B}, L={L}, min_id={min_id}, max_id={max_id}, vocab_size={vocab_size}")
        
        if max_id >= vocab_size:
            raise ValueError(
                f"Invalid input_ids: max_id={max_id} >= vocab_size={vocab_size}. "
                f"min_id={min_id}, max_id={max_id}. "
                f"First sequence sample: {seqs[0][:50] if seqs else 'N/A'}"
            )
        
        # Additional check: ensure no invalid tokens (UNK, etc.)
        # ESMFold tokenizer should map amino acids correctly
        if tokenizer.unk_token_id is not None:
            unk_count = (input_ids == tokenizer.unk_token_id).sum().item()
            if unk_count > 0:
                print(f"Warning: Found {unk_count} UNK tokens in batch")

        # Explicit position_ids (recommended for ESMFold)
        B_actual, L_actual = batch["input_ids"].shape
        batch["position_ids"] = torch.arange(L_actual, device=cfg.device).unsqueeze(0).expand(B_actual, L_actual)

        # Approx token count
        tokens = int(batch["attention_mask"].sum().item()) if "attention_mask" in batch else L_actual * B_actual

        # --- Baseline timing (stock model) ---
        t_base = time_it_cuda(
            lambda: forward_esmfold(model, batch, cfg.num_recycles),
            cfg.iters,
            cfg.warmup
        )

        # --- Patched timing (fused MLP1 in stem) ---
        originals = patch_esmfold_stem_mlp1(model)

        # Trigger Triton JIT/autotune once before timing
        _ = forward_esmfold(model, batch, cfg.num_recycles)
        torch.cuda.synchronize()

        t_pat = time_it_cuda(
            lambda: forward_esmfold(model, batch, cfg.num_recycles),
            cfg.iters,
            cfg.warmup
        )

        # --- Correctness check ---
        # Restore baseline and compute reference output
        unpatch_esmfold_stem_mlp1(model, originals)
        y_ref = forward_esmfold(model, batch, cfg.num_recycles)

        # Re-patch and compute patched output
        originals = patch_esmfold_stem_mlp1(model)
        y_pat = forward_esmfold(model, batch, cfg.num_recycles)
        unpatch_esmfold_stem_mlp1(model, originals)

        max_abs = (y_ref - y_pat).abs().max().item()
        mean_abs = (y_ref - y_pat).abs().mean().item()

        speedup = t_base / t_pat if t_pat > 0 else float("inf")
        tok_per_s_base = tokens / (t_base / 1e3) if t_base > 0 else 0
        tok_per_s_pat = tokens / (t_pat / 1e3) if t_pat > 0 else 0

        results.append({
            "benchmark_name": "esmfold_synth",
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
                "num_recycles": cfg.num_recycles,
            }
        })

    return results


def main():
    results = run_benchmark()
    cfg = BenchCfg()
    
    print(f"Model: {cfg.model_id} | dtype: {cfg.dtype} | device: {cfg.device}")
    print(f"Iters: {cfg.iters}, warmup: {cfg.warmup}, num_recycles: {cfg.num_recycles}")
    print("Benchmark: Baseline vs Patched (fused MLP1 in ESM stem)\n")

    for result in results:
        meta = result["metadata"]
        print(f"B={meta['B']} L={meta['L']} | tokens={meta['tokens']}")
        print(f"  baseline: {result['baseline_time_ms']:.3f} ms/iter  ({meta['tokens_per_sec_baseline']:,.0f} tokens/s)")
        print(f"  patched : {result['triton_time_ms']:.3f} ms/iter  ({meta['tokens_per_sec_triton']:,.0f} tokens/s)")
        print(f"  speedup : {result['speedup']:.2f}x" if result['speedup'] else "  speedup : inf")
        print(f"  max|diff|: {result['max_diff']:.3e}  mean|diff|: {result['mean_diff']:.3e}\n")


if __name__ == "__main__":
    main()
