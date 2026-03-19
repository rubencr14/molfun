---
title: Tutorials
---

# Tutorials

Hands-on guides that walk you through real-world protein modeling tasks with Molfun.
Each tutorial is self-contained and includes runnable code.

---

<div class="tutorial-grid" markdown>

<div class="tutorial-card" markdown>
:material-thermometer:{ .lg } **[Stability Prediction](stability-prediction.md)**
{ .tutorial-card-title }

Predict protein thermostability from sequence using a **HeadOnly** fine-tuning strategy. Load a CSV of sequences with stability labels, train, and evaluate with a scatter plot.

<span class="badge badge-beginner">Beginner</span> <span class="badge">~20 min</span>
</div>

<div class="tutorial-card" markdown>
:material-link-variant:{ .lg } **[Binding Affinity Prediction](binding-affinity.md)**
{ .tutorial-card-title }

Fetch PDBbind affinity data, apply **LoRA** fine-tuning, and predict binding affinity (Kd/Ki). Compare LoRA against full fine-tuning on a real benchmark.

<span class="badge badge-intermediate">Intermediate</span> <span class="badge">~30 min</span>
</div>

<div class="tutorial-card" markdown>
:material-dna:{ .lg } **[Kinase Structure Refinement](kinase-refinement.md)**
{ .tutorial-card-title }

Use the built-in `kinases_human` collection with **PartialFinetune** to refine predicted kinase structures using FAPE loss and evaluate structural quality.

<span class="badge badge-intermediate">Intermediate</span> <span class="badge">~30 min</span>
</div>

<div class="tutorial-card" markdown>
:material-scale-balance:{ .lg } **[LoRA for Small Datasets](lora-small-datasets.md)**
{ .tutorial-card-title }

When you only have ~50 proteins, full fine-tuning overfits. Learn when to use **LoRA** vs **HeadOnly**, compare overfitting curves, and tune rank and alpha.

<span class="badge badge-beginner">Beginner</span> <span class="badge">~15 min</span>
</div>

<div class="tutorial-card" markdown>
:material-puzzle-edit:{ .lg } **[Building Custom Architectures](custom-architectures.md)**
{ .tutorial-card-title }

Compose models from scratch with **ModelBuilder**: choose embedders, transformer blocks, and structure modules. Train a fully custom architecture on structure prediction.

<span class="badge badge-advanced">Advanced</span> <span class="badge">~45 min</span>
</div>

<div class="tutorial-card" markdown>
:material-chart-timeline-variant:{ .lg } **[Experiment Tracking](experiment-tracking.md)**
{ .tutorial-card-title }

Set up **WandB**, **Comet**, or **MLflow** tracking in one line. Use `CompositeTracker` to log to multiple backends simultaneously and compare runs.

<span class="badge badge-beginner">Beginner</span> <span class="badge">~15 min</span>
</div>

<div class="tutorial-card" markdown>
:material-pipe:{ .lg } **[YAML Pipelines](yaml-pipelines.md)**
{ .tutorial-card-title }

Define reproducible end-to-end workflows with `Pipeline.from_yaml()`. Fetch data, preprocess, train, and evaluate --- all from a single YAML recipe file.

<span class="badge badge-intermediate">Intermediate</span> <span class="badge">~20 min</span>
</div>

</div>

---

## Learning Path

If you are new to Molfun, we recommend following the tutorials in this order:

```mermaid
graph LR
    A["Stability Prediction<br/><small>HeadOnly basics</small>"] --> B["LoRA for Small Datasets<br/><small>PEFT fundamentals</small>"]
    B --> C["Binding Affinity<br/><small>LoRA + real data</small>"]
    C --> D["Kinase Refinement<br/><small>Partial fine-tune</small>"]
    D --> E["Custom Architectures<br/><small>ModelBuilder</small>"]

    F["Experiment Tracking<br/><small>Any time</small>"] -.-> A
    G["YAML Pipelines<br/><small>After any tutorial</small>"] -.-> C

    style A fill:#40a02b,stroke:#40a02b,color:#ffffff
    style B fill:#40a02b,stroke:#40a02b,color:#ffffff
    style C fill:#df8e1d,stroke:#df8e1d,color:#ffffff
    style D fill:#df8e1d,stroke:#df8e1d,color:#ffffff
    style E fill:#d20f39,stroke:#d20f39,color:#ffffff
    style F fill:#209fb5,stroke:#209fb5,color:#ffffff
    style G fill:#209fb5,stroke:#209fb5,color:#ffffff
```

!!! tip "Prerequisites"

    All tutorials assume you have Molfun installed. If you have not done so yet, follow
    the [Installation guide](../getting-started/installation.md) first.
