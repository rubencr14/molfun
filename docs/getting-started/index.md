---
title: Getting Started
---

# Getting Started

**Welcome to Molfun.** These guides walk you through everything you need to go from zero to
fine-tuning protein structure models --- whether you are a computational biologist running your
first prediction or an ML engineer integrating Molfun into a production pipeline.

!!! tip "No prior experience with AlphaFold or OpenFold required"

    The guides assume familiarity with Python and basic machine learning concepts.
    Knowledge of protein structure prediction is helpful but not necessary --- we explain
    the key ideas as we go.

---

## Learning Path

Follow the guides in order for the smoothest experience, or jump to whichever step
matches where you are.

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
<span class="feature-icon">:material-download:</span>

### 1. Installation

Set up Python, install Molfun and its optional extras, and verify everything works.

[:octicons-arrow-right-24: Install Molfun](installation.md)
</div>

<div class="feature-card" markdown>
<span class="feature-icon">:material-rocket-launch:</span>

### 2. Quick Start

Three code snippets to see what Molfun can do: structure prediction, LoRA fine-tuning,
and the CLI --- all in under five minutes.

[:octicons-arrow-right-24: Quick Start](quickstart.md)
</div>

<div class="feature-card" markdown>
<span class="feature-icon">:material-molecule:</span>

### 3. First Prediction

A step-by-step tutorial that loads a pretrained model, predicts a 3D structure, explores the
output tensors, saves a PDB file, and visualizes the result.

[:octicons-arrow-right-24: First Prediction](first-prediction.md)
</div>

<div class="feature-card" markdown>
<span class="feature-icon">:material-tune:</span>

### 4. First Fine-Tuning

End-to-end LoRA fine-tuning: prepare a dataset, pick a strategy, run `model.fit()`, track
the experiment, and evaluate the result.

[:octicons-arrow-right-24: First Fine-Tuning](first-finetuning.md)
</div>

</div>

---

## What Comes Next?

Once you have completed these guides you will be able to:

- [x] Predict protein structures, properties, and binding affinities
- [x] Fine-tune models with LoRA, head-only, partial, or full strategies
- [x] Save, load, and export models for production

From here, explore the **[Tutorials](../tutorials/index.md)** for real-world workflows,
the **[Architecture](../architecture/index.md)** docs to understand how the pieces fit together,
or the **[API Reference](../reference/index.md)** for complete method signatures.
