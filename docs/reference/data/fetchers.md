# Data Fetchers

Fetchers download biological data from external sources: PDB structures,
binding affinity datasets, and multiple sequence alignments.

## Quick Start

```python
from molfun.data.sources import PDBFetcher, AffinityFetcher, MSAProvider

# Fetch PDB structures
fetcher = PDBFetcher()
structures = fetcher.fetch(["1ABC", "2DEF", "3GHI"])

# Fetch affinity data
affinity = AffinityFetcher()
data = affinity.fetch(source="pdbbind", version="2020")

# Generate MSAs
msa = MSAProvider(backend="colabfold")
alignment = msa.fetch("MKFLILLFNILCLFPVLAADNH...")
```

## PDBFetcher

::: molfun.data.sources.pdb.PDBFetcher
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Download PDB/mmCIF files from the RCSB PDB.

### fetch

```python
fetcher = PDBFetcher(cache_dir="./pdb_cache", format="mmcif")

# Fetch by PDB IDs
structures = fetcher.fetch(["1ABC", "2DEF"])

# Fetch with filters
structures = fetcher.fetch(
    ids=None,
    resolution_max=2.5,
    organism="Homo sapiens",
    method="X-RAY DIFFRACTION",
    max_results=100,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_dir` | `str \| Path` | `"~/.molfun/pdb"` | Local cache directory |
| `format` | `str` | `"mmcif"` | File format: `"pdb"` or `"mmcif"` |

#### fetch() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ids` | `list[str] \| None` | `None` | PDB IDs to download |
| `resolution_max` | `float \| None` | `None` | Maximum resolution in Angstroms |
| `organism` | `str \| None` | `None` | Source organism filter |
| `method` | `str \| None` | `None` | Experimental method filter |
| `max_results` | `int \| None` | `None` | Limit number of results (for queries) |

**Returns:** `list[Path]` of downloaded file paths.

---

## AffinityFetcher

::: molfun.data.sources.affinity.AffinityFetcher
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Download protein-ligand binding affinity datasets (e.g., PDBbind).

### fetch

```python
fetcher = AffinityFetcher(cache_dir="./affinity_cache")

data = fetcher.fetch(
    source="pdbbind",
    version="2020",
    split="refined",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_dir` | `str \| Path` | `"~/.molfun/affinity"` | Local cache directory |

#### fetch() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | `"pdbbind"` | Data source name |
| `version` | `str` | `"2020"` | Dataset version |
| `split` | `str` | `"refined"` | Dataset split: `"general"`, `"refined"`, `"core"` |

**Returns:** `AffinityDataset` ready for training.

---

## MSAProvider

::: molfun.data.sources.msa.MSAProvider
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Generate or fetch multiple sequence alignments for protein sequences.

### fetch

```python
msa = MSAProvider(backend="colabfold", cache_dir="./msa_cache")

# Single sequence
alignment = msa.fetch("MKFLILLFNILCLFPVLAADNH...")

# Batch
alignments = msa.fetch_batch(
    ["MKFLILLFNILCLFPVLAADNH...", "MKTAYIAKQRQISFVKSH..."],
    num_workers=4,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"colabfold"` | MSA generation backend |
| `cache_dir` | `str \| Path` | `"~/.molfun/msa"` | Cache directory for alignments |
| `database` | `str` | `"uniref30"` | Sequence database to search |

#### fetch() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequence` | `str` | *required* | Amino acid sequence |
| `max_seqs` | `int` | `512` | Maximum number of sequences in MSA |

**Returns:** `str` -- path to the generated A3M file.
