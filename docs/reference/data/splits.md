# DataSplitter

Utilities for splitting datasets into train/validation/test sets using
different strategies appropriate for biological data.

## Quick Start

```python
from molfun.data.splits import DataSplitter

splitter = DataSplitter()

# Random split
train, val, test = splitter.random(dataset, fractions=[0.8, 0.1, 0.1])

# Temporal split (by deposition date)
train, val, test = splitter.temporal(dataset, cutoff_date="2020-05-01")

# Identity-based split (sequence clustering)
train, val, test = splitter.identity(dataset, threshold=0.3)
```

## Class Reference

::: molfun.data.splits.DataSplitter
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### random

Standard random split.

```python
train, val, test = splitter.random(
    dataset,
    fractions=[0.8, 0.1, 0.1],
    seed=42,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `Dataset` | *required* | Dataset to split |
| `fractions` | `list[float]` | `[0.8, 0.1, 0.1]` | Train/val/test fractions (must sum to 1.0) |
| `seed` | `int` | `42` | Random seed for reproducibility |

**Returns:** Tuple of `(train_dataset, val_dataset, test_dataset)`.

---

### temporal

Split by deposition date to prevent data leakage from future structures.

```python
train, val, test = splitter.temporal(
    dataset,
    cutoff_date="2020-05-01",
    val_cutoff_date="2021-01-01",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `Dataset` | *required* | Dataset to split (must have date metadata) |
| `cutoff_date` | `str` | *required* | Train/val boundary date (ISO format) |
| `val_cutoff_date` | `str \| None` | `None` | Val/test boundary date. If `None`, remaining data is split 50/50. |

**Returns:** Tuple of `(train_dataset, val_dataset, test_dataset)`.

---

### identity

Cluster sequences by identity and split at the cluster level. Ensures
no test protein is similar to any training protein above the threshold.

```python
train, val, test = splitter.identity(
    dataset,
    threshold=0.3,
    fractions=[0.8, 0.1, 0.1],
    seed=42,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `Dataset` | *required* | Dataset to split (must have sequence data) |
| `threshold` | `float` | `0.3` | Maximum sequence identity between splits (0.0--1.0) |
| `fractions` | `list[float]` | `[0.8, 0.1, 0.1]` | Approximate train/val/test fractions |
| `seed` | `int` | `42` | Random seed |

**Returns:** Tuple of `(train_dataset, val_dataset, test_dataset)`.

---

## Choosing a Split Strategy

| Strategy | When to Use | Prevents |
|----------|-------------|----------|
| **random** | Quick experiments, baselines | Nothing (data leakage possible) |
| **temporal** | Realistic evaluation, time-series data | Temporal leakage |
| **identity** | Generalization to novel proteins | Homology leakage |

For publication-quality results, prefer **identity** splits with a threshold
of 0.3 (30% sequence identity), which is the standard in protein structure
prediction benchmarks (e.g., CASP, CAMEO).
