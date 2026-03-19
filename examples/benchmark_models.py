#!/usr/bin/env python
"""
Benchmark a custom fine-tuned model against AlphaFold 3 on structure quality
and binding affinity prediction — the workflow behind a publication table.

Scenario
--------
You have fine-tuned a Pairformer + IPA model with gated attention and LoRA
on a kinase dataset.  Before submitting a paper you need to demonstrate that
it outperforms (or matches) AlphaFold 3 on standard benchmarks, and you need
publication-ready tables in LaTeX and Markdown.

This script:
  1. Loads both models (your checkpoint + AF3 baseline).
  2. Evaluates them on the same benchmark suites with structural and
     affinity metrics.
  3. Profiles inference latency and peak VRAM.
  4. Aggregates results in a Leaderboard.
  5. Exports Markdown, LaTeX, and JSON.

Requires:
  pip install molfun[openfold]

Typical runtime: ~5-20 min on A100 depending on dataset size.
"""

from pathlib import Path

import torch

# ── Benchmarking imports ──────────────────────────────────────────────

from molfun.benchmarks import (
    # Metrics (used directly when you need fine-grained control)
    create_metrics,
    MetricCollection,
    # Suites & evaluator
    BenchmarkTask,
    BenchmarkSuite,
    TaskType,
    ModelEvaluator,
    # Reports & comparison
    BenchmarkReport,
    Leaderboard,
    # Performance profiling
    InferenceBenchmark,
)

# ── Model imports ─────────────────────────────────────────────────────

from molfun.models.structure import MolfunStructureModel
from molfun.modules.builder import ModelBuilder


# ── Helper to create simulated results (used in sections 3–5) ────────

def _make_result(task_name: str, n: int = 100, **metrics) -> "TaskResult":
    from molfun.benchmarks.report import TaskResult
    return TaskResult(task_name=task_name, metrics=metrics, n_samples=n, duration_s=30.0)


# =====================================================================
# 1. LOAD / BUILD BOTH MODELS
# =====================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_SINGLE, D_PAIR = 256, 128

print("=" * 70)
print("  MODEL BENCHMARK: Custom Gated-Pairformer vs AlphaFold 3 baseline")
print("=" * 70)

# ── Model A: your custom architecture ────────────────────────────────

print("\n[1/6] Building custom model...")

# In a real run you would load your trained checkpoint:
#   model_custom = MolfunStructureModel.load("runs/kinase_lora/checkpoint", device=DEVICE)
#
# For this example, we build from scratch to keep it self-contained:
builder = ModelBuilder(
    embedder="input",
    embedder_config={"d_single": D_SINGLE, "d_pair": D_PAIR, "d_msa": D_SINGLE},
    block="pairformer",
    block_config={
        "d_single": D_SINGLE,
        "d_pair": D_PAIR,
        "n_heads": 8,
        "attention_cls": "gated",
    },
    n_blocks=8,
    structure_module="ipa",
    structure_module_config={"d_single": D_SINGLE, "d_pair": D_PAIR},
)
built = builder.build()

model_custom = MolfunStructureModel.from_custom(
    built,
    device=DEVICE,
    head="affinity",
    head_config={"single_dim": D_SINGLE, "hidden_dim": 64},
)
print(f"  Custom: {built.param_summary()}")

# ── Model B: AlphaFold 3 / OpenFold baseline ────────────────────────

# In a real run:
#   from openfold.config import model_config
#   config = model_config("model_1_ptm")
#   model_af3 = MolfunStructureModel(
#       "openfold", config=config,
#       weights=str(Path.home() / ".molfun/weights/finetuning_ptm_2.pt"),
#       device=DEVICE,
#       head="affinity",
#       head_config={"single_dim": 384, "hidden_dim": 128},
#   )
#
# For demonstration, build a second architecture that represents the baseline:
builder_baseline = ModelBuilder(
    embedder="input",
    embedder_config={"d_single": D_SINGLE, "d_pair": D_PAIR, "d_msa": D_SINGLE},
    block="pairformer",
    block_config={
        "d_single": D_SINGLE,
        "d_pair": D_PAIR,
        "n_heads": 8,
        "attention_cls": "flash",
    },
    n_blocks=8,
    structure_module="ipa",
    structure_module_config={"d_single": D_SINGLE, "d_pair": D_PAIR},
)

model_baseline = MolfunStructureModel.from_custom(
    builder_baseline.build(),
    device=DEVICE,
    head="affinity",
    head_config={"single_dim": D_SINGLE, "hidden_dim": 64},
)
print(f"  Baseline (AF3-style): {builder_baseline.build().param_summary()}")


# =====================================================================
# 2. DEFINE THE BENCHMARK SUITE
# =====================================================================

print("\n[2/6] Configuring benchmark suite...")

# Standard PDBbind suite for affinity:
# suite_affinity = BenchmarkSuite.pdbbind(
#     index_path="data/pdbbind/INDEX_refined_data.2020",
#     structures_dir="data/pdbbind/structures",
# )

# Structure quality suite:
# suite_structure = BenchmarkSuite.structure_quality(
#     targets_dir="data/casp15/targets",
# )

# For demonstration we create a custom suite that can run without data:
suite = BenchmarkSuite.custom(
    name="KinaseAffinity-v1",
    description="Kinase binding affinity + structure quality benchmark",
    tasks=[
        BenchmarkTask(
            name="kinase_affinity",
            data_source="data/kinase_benchmark",
            split="identity_30",
            metrics=("mae", "rmse", "pearson", "spearman", "r2"),
            task_type=TaskType.REGRESSION,
            description="pKd regression on 285 kinase complexes",
        ),
        BenchmarkTask(
            name="kinase_structure",
            data_source="data/kinase_benchmark/structures",
            split="all",
            metrics=("gdt_ts", "tm_score", "lddt", "coord_rmsd"),
            task_type=TaskType.STRUCTURE,
            description="Structure quality on kinase test targets",
        ),
    ],
)

print(f"  {suite.summary()}")


# =====================================================================
# 3. EVALUATE BOTH MODELS
# =====================================================================

print("\n[3/6] Evaluating models on benchmark suite...")

# In a real run the evaluator finds and loads the data automatically:
# report_custom = ModelEvaluator(model_custom, suite, device=DEVICE).run()
# report_baseline = ModelEvaluator(model_baseline, suite, device=DEVICE).run()
#
# Since we have no actual data, we simulate reports to show the full
# output pipeline:

report_custom = BenchmarkReport(
    model_name="GatedPairformer-LoRA-r8",
    suite_name=suite.name,
    results={
        "kinase_affinity": _make_result("kinase_affinity", mae=0.82, rmse=1.05,
                                        pearson=0.71, spearman=0.69, r2=0.50, n=285),
        "kinase_structure": _make_result("kinase_structure", gdt_ts=72.3,
                                         tm_score=0.81, lddt=68.5, coord_rmsd=2.1, n=50),
    },
    total_duration_s=245.3,
    metadata={"device": DEVICE, "lora_rank": 8, "attention": "gated"},
)

report_baseline = BenchmarkReport(
    model_name="AlphaFold3-Baseline",
    suite_name=suite.name,
    results={
        "kinase_affinity": _make_result("kinase_affinity", mae=0.95, rmse=1.22,
                                        pearson=0.64, spearman=0.61, r2=0.41, n=285),
        "kinase_structure": _make_result("kinase_structure", gdt_ts=68.1,
                                         tm_score=0.77, lddt=64.2, coord_rmsd=2.6, n=50),
    },
    total_duration_s=312.7,
    metadata={"device": DEVICE, "attention": "flash"},
)

print(f"  Custom model evaluated in {report_custom.total_duration_s:.0f}s")
print(f"  Baseline evaluated in {report_baseline.total_duration_s:.0f}s")


# =====================================================================
# 4. INFERENCE PERFORMANCE COMPARISON
# =====================================================================

print("\n[4/6] Profiling inference performance...")

# bench_custom = InferenceBenchmark(model_custom, device=DEVICE, warmup=3, repeats=10)
# bench_baseline = InferenceBenchmark(model_baseline, device=DEVICE, warmup=3, repeats=10)
# perf_custom = bench_custom.run(seq_lengths=[128, 256, 512])
# perf_baseline = bench_baseline.run(seq_lengths=[128, 256, 512])
# print(perf_custom.to_markdown())
# print(perf_baseline.to_markdown())

print("  (Skipped in demo — uncomment with real models + GPU)")


# =====================================================================
# 5. BUILD LEADERBOARD
# =====================================================================

print("\n[5/6] Building leaderboard...")

leaderboard = Leaderboard()
leaderboard.add(report_custom)
leaderboard.add(report_baseline)

# Rank by affinity MAE (lower is better)
print("\n  Affinity ranking (MAE ↓):")
for rank, (name, val) in enumerate(leaderboard.rank("kinase_affinity", "mae", ascending=True), 1):
    marker = " ← best" if rank == 1 else ""
    print(f"    {rank}. {name}: {val:.4f}{marker}")

# Rank by structure TM-score (higher is better)
print("\n  Structure ranking (TM-score ↑):")
for rank, (name, val) in enumerate(leaderboard.rank("kinase_structure", "tm_score", ascending=False), 1):
    marker = " ← best" if rank == 1 else ""
    print(f"    {rank}. {name}: {val:.4f}{marker}")


# =====================================================================
# 6. EXPORT FOR PUBLICATION
# =====================================================================

print("\n[6/6] Exporting results...")

output_dir = Path("runs/benchmark_results")
output_dir.mkdir(parents=True, exist_ok=True)

# Markdown — for README / supplementary
md_custom = report_custom.to_markdown()
md_baseline = report_baseline.to_markdown()
comparison_md = leaderboard.table("kinase_affinity", metrics=["mae", "rmse", "pearson", "spearman", "r2"])

(output_dir / "results_custom.md").write_text(md_custom)
(output_dir / "results_baseline.md").write_text(md_baseline)
(output_dir / "comparison_affinity.md").write_text(
    "## Affinity Benchmark: Model Comparison\n\n" + comparison_md
)

# LaTeX — copy-paste into your paper
latex_custom = report_custom.to_latex()
(output_dir / "table_custom.tex").write_text(latex_custom)

latex_baseline = report_baseline.to_latex()
(output_dir / "table_baseline.tex").write_text(latex_baseline)

# JSON — machine-readable for further analysis
(output_dir / "report_custom.json").write_text(report_custom.to_json())
(output_dir / "report_baseline.json").write_text(report_baseline.to_json())

# Leaderboard persistence
leaderboard.save(str(output_dir / "leaderboard.json"))

print(f"\n  Saved to {output_dir}/:")
print(f"    - results_custom.md, results_baseline.md")
print(f"    - comparison_affinity.md")
print(f"    - table_custom.tex, table_baseline.tex")
print(f"    - report_custom.json, report_baseline.json")
print(f"    - leaderboard.json")

# Show the comparison table
print("\n" + "=" * 70)
print("  COMPARISON TABLE (Markdown)")
print("=" * 70)
print()
print("### Affinity Prediction")
print()
print(comparison_md)
print()
print("### Structure Quality")
print()
print(leaderboard.table("kinase_structure", metrics=["gdt_ts", "tm_score", "lddt", "coord_rmsd"]))

# Show LaTeX for paper
print()
print("=" * 70)
print("  LaTeX TABLE (copy into your .tex)")
print("=" * 70)
print()
print(latex_custom)

print()
print("=" * 70)
print("  CONCLUSION")
print("=" * 70)

# Automated decision
custom_mae = report_custom.metric("kinase_affinity", "mae")
baseline_mae = report_baseline.metric("kinase_affinity", "mae")
custom_tm = report_custom.metric("kinase_structure", "tm_score")
baseline_tm = report_baseline.metric("kinase_structure", "tm_score")

print()
if custom_mae < baseline_mae and custom_tm > baseline_tm:
    print("  Result: Custom model OUTPERFORMS baseline on both tasks.")
    print("  Ready for publication.")
elif custom_mae < baseline_mae:
    print("  Result: Custom model is better on affinity, comparable on structure.")
    print("  Consider additional structure evaluation before publication.")
elif custom_tm > baseline_tm:
    print("  Result: Custom model is better on structure, weaker on affinity.")
    print("  Consider ensemble approaches or further affinity fine-tuning.")
else:
    print("  Result: Baseline still wins.  Back to the drawing board.")

print()
print("  To push the winning model to Hugging Face Hub:")
print("    model.push_to_hub('your-org/kinase-gated-pairformer',")
print("                      metrics=report.all_metrics())")


