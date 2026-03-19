# Collections

Curated dataset collections for common benchmarks and training sets.
Collections provide a convenient way to fetch well-known datasets with
a single function call.

## Quick Start

```python
from molfun.data.collections import fetch_collection, list_collections

# See what's available
print(list_collections())
# ["casp14", "casp15", "cameo", "pdbbind_refined", ...]

# Fetch a collection
dataset = fetch_collection("casp15", split="test")

# Check collection size before downloading
from molfun.data.collections import count_collection
n = count_collection("pdbbind_refined")
print(f"PDBbind refined set: {n} complexes")
```

## Functions

### fetch_collection

::: molfun.data.collections.fetch_collection
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

Download and return a curated dataset collection.

```python
from molfun.data.collections import fetch_collection

# Structure prediction benchmark
ds = fetch_collection("casp15", split="test", cache_dir="./data")

# Affinity dataset
ds = fetch_collection("pdbbind_refined", version="2020")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Collection name |
| `split` | `str \| None` | `None` | Dataset split (collection-dependent) |
| `version` | `str \| None` | `None` | Dataset version |
| `cache_dir` | `str \| Path` | `"~/.molfun/collections"` | Download cache directory |

**Returns:** `Dataset` appropriate for the collection type.

---

### count_collection

::: molfun.data.collections.count_collection
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

Return the number of entries in a collection without downloading it.

```python
from molfun.data.collections import count_collection

n = count_collection("casp15")
print(f"CASP15 has {n} targets")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Collection name |

**Returns:** `int`

---

### list_collections

::: molfun.data.collections.list_collections
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

List all available collection names.

```python
from molfun.data.collections import list_collections

for name in list_collections():
    n = count_collection(name)
    print(f"{name}: {n} entries")
```

**Returns:** `list[str]`

---

## CollectionSpec

::: molfun.data.collections.CollectionSpec
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Dataclass describing a collection's metadata.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Collection identifier |
| `description` | `str` | Human-readable description |
| `task` | `str` | Task type: `"structure"`, `"affinity"`, `"property"` |
| `size` | `int` | Number of entries |
| `url` | `str` | Source URL |
| `citation` | `str` | BibTeX citation key |

---

## Available Collections

| Name | Task | Size | Description |
|------|------|------|-------------|
| `casp14` | structure | 87 | CASP14 free-modeling targets |
| `casp15` | structure | 109 | CASP15 free-modeling targets |
| `cameo` | structure | ~500/yr | CAMEO weekly structure prediction targets |
| `pdbbind_refined` | affinity | ~5,300 | PDBbind refined set |
| `pdbbind_core` | affinity | ~290 | PDBbind core set (test benchmark) |
| `pdbbind_general` | affinity | ~19,400 | PDBbind general set |
