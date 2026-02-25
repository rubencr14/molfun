#!/usr/bin/env python
"""
Autonomous architecture search with an LLM agent.

The agent iteratively:
  1. Proposes architectural modifications (attention type, depth, hidden dims…)
  2. Trains each variant on your data
  3. Evaluates and records results
  4. Reasons about what worked and plans the next experiment

You can run this on a mini PC 24/7 with LM Studio (local LLM) while the
training runs on a remote GPU box, or let it talk to Claude/GPT-4 APIs.

Requires:
  pip install molfun[agents]
  # For local LLMs: pip install molfun[agents-ollama] or use LM Studio

Config:
  Set OPENAI_API_KEY for GPT-4, ANTHROPIC_API_KEY for Claude,
  or point to a local LM Studio / Ollama endpoint.
"""

import os
from pathlib import Path

from molfun.agents.researcher import ResearchAgent
from molfun.agents.experiment import ExperimentConfig
from molfun.agents.llm import OllamaBackend, OpenAIBackend

# ── 1. Choose your LLM backend ───────────────────────────────────────

# Option A: Local LLM via LM Studio (free, private, runs on your mini PC)
# LM Studio exposes an OpenAI-compatible API on localhost:
#
# llm = OpenAIBackend(
#     model="local-model",
#     api_key="lm-studio",
#     base_url="http://192.168.1.42:1234/v1",  # your LM Studio IP
# )

# Option B: Local Ollama
# llm = OllamaBackend(model="llama3.1:70b", host="http://localhost:11434")

# Option C: Cloud API (most capable)
if os.getenv("ANTHROPIC_API_KEY"):
    from molfun.agents.llm import AnthropicBackend
    llm = AnthropicBackend(model="claude-sonnet-4-20250514")
    print("Using Claude")
elif os.getenv("OPENAI_API_KEY"):
    llm = OpenAIBackend(model="gpt-4o")
    print("Using GPT-4o")
else:
    llm = OllamaBackend(model="llama3.1:8b")
    print("Using local Ollama (llama3.1:8b)")

# ── 2. Define the search space ───────────────────────────────────────

search_config = ExperimentConfig(
    name="kinase_arch_search",
    model_type="openfold",
    description=(
        "Search for the best architecture to predict binding affinity "
        "on kinase targets. Explore attention types (flash, gated, linear), "
        "trunk depth (4-12 blocks), LoRA rank (4-16), and structure module "
        "(IPA vs diffusion). Training budget: 10 epochs per trial."
    ),
    search_space={
        "attention_cls": ["flash", "gated", "linear"],
        "n_blocks": [4, 6, 8, 12],
        "lora_rank": [4, 8, 16],
        "structure_module": ["ipa", "diffusion"],
        "lr": [1e-4, 5e-4, 1e-3],
    },
    objective="minimize val_mae",
    max_trials=20,
    data_config={
        "pdb_ids_file": "data/kinase_ids.txt",
        "affinity_csv": "data/kinase_affinities.csv",
        "msa_mode": "single",
        "max_seq_len": 256,
    },
    training_config={
        "epochs": 10,
        "batch_size": 1,
        "strategy": "lora",
        "early_stopping_patience": 3,
    },
)

# ── 3. Launch the agent ──────────────────────────────────────────────

agent = ResearchAgent(
    llm=llm,
    config=search_config,
    output_dir="runs/kinase_arch_search",
    memory_path="runs/kinase_arch_search/memory.json",
)

print(f"\nStarting architecture search: {search_config.name}")
print(f"  Trials: up to {search_config.max_trials}")
print(f"  Objective: {search_config.objective}")
print(f"  Output: runs/kinase_arch_search/")
print()

# Run the full search loop
# The agent will:
#   - Pick hyperparams based on past results
#   - Train each variant
#   - Log reasoning to memory (inspectable in memory.json)
#   - Stop early if convergence is detected
journal = agent.run()

# ── 4. Inspect results ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("SEARCH COMPLETE")
print("=" * 60)

# Best trial
best = journal.best_trial()
if best:
    print(f"\nBest trial: {best.trial_id}")
    print(f"  Config:  {best.config}")
    print(f"  Metrics: {best.metrics}")
    checkpoint = Path("runs/kinase_arch_search") / best.trial_id / "checkpoint"
    print(f"  Checkpoint: {checkpoint}")

# All results
print(f"\nAll {len(journal.trials)} trials:")
for trial in sorted(journal.trials, key=lambda t: t.metrics.get("val_mae", 999)):
    print(f"  {trial.trial_id}: val_mae={trial.metrics.get('val_mae', 'N/A'):.4f} "
          f"| {trial.config}")

# Agent reasoning (useful for papers / understanding what it learned)
print(f"\nAgent reasoning log saved to: runs/kinase_arch_search/memory.json")
print("Review it to understand the agent's hypotheses and decisions.")
