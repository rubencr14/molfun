# Datasets

PyTorch `Dataset` implementations for structure prediction, affinity
prediction, and streaming workloads.

## Quick Start

```python
from molfun.data.datasets import StructureDataset, AffinityDataset

# Structure dataset from PDB files
ds = StructureDataset(
    data_dir="./pdb_files",
    max_length=512,
)

# Affinity dataset
ds = AffinityDataset(
    csv_path="./affinity_data.csv",
    structure_dir="./structures",
    ligand_dir="./ligands",
)

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=2, num_workers=4)
```

## StructureDataset

::: molfun.data.datasets.structure.StructureDataset
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Dataset for protein structure prediction training. Loads PDB/mmCIF files
and produces feature dictionaries compatible with the model forward pass.

```python
from molfun.data.datasets import StructureDataset

ds = StructureDataset(
    data_dir="./pdb_files",
    max_length=512,
    crop_strategy="contiguous",
    format="mmcif",
)

sample = ds[0]
# {
#     "aatype": Tensor (L,),
#     "residue_index": Tensor (L,),
#     "all_atom_positions": Tensor (L, 37, 3),
#     "all_atom_mask": Tensor (L, 37),
#     ...
# }
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | `str \| Path` | *required* | Directory containing structure files |
| `max_length` | `int` | `512` | Maximum sequence length (longer chains are cropped) |
| `crop_strategy` | `str` | `"contiguous"` | Cropping: `"contiguous"`, `"random"`, `"spatial"` |
| `format` | `str` | `"mmcif"` | Input format: `"pdb"` or `"mmcif"` |
| `msa_dir` | `str \| None` | `None` | Directory with precomputed MSA files |
| `transform` | `callable \| None` | `None` | Optional transform applied to each sample |

---

## AffinityDataset

::: molfun.data.datasets.affinity.AffinityDataset
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Dataset for protein-ligand binding affinity prediction.

```python
from molfun.data.datasets import AffinityDataset

ds = AffinityDataset(
    csv_path="./affinity_labels.csv",
    structure_dir="./structures",
    ligand_dir="./ligands",
    label_column="pKd",
)

sample = ds[0]
# {
#     "aatype": Tensor,
#     "residue_index": Tensor,
#     "ligand_features": Tensor,
#     "affinity_label": Tensor,
#     ...
# }
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path` | `str \| Path` | *required* | CSV with PDB IDs and affinity labels |
| `structure_dir` | `str \| Path` | *required* | Directory with protein structure files |
| `ligand_dir` | `str \| Path` | *required* | Directory with ligand files (SDF) |
| `label_column` | `str` | `"affinity"` | Column name for affinity labels |
| `max_length` | `int` | `512` | Maximum protein sequence length |
| `transform` | `callable \| None` | `None` | Optional transform |

---

## StreamingStructureDataset

::: molfun.data.datasets.streaming.StreamingStructureDataset
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

An `IterableDataset` for large-scale training that streams data from
disk without loading everything into memory.

```python
from molfun.data.datasets import StreamingStructureDataset

ds = StreamingStructureDataset(
    data_dir="./large_pdb_dataset",
    max_length=512,
    shuffle_buffer=1000,
)

loader = DataLoader(ds, batch_size=2, num_workers=4)
for batch in loader:
    ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | `str \| Path` | *required* | Directory containing structure files |
| `max_length` | `int` | `512` | Maximum sequence length |
| `shuffle_buffer` | `int` | `1000` | Buffer size for streaming shuffle |
| `format` | `str` | `"mmcif"` | Input format |
| `seed` | `int \| None` | `None` | Random seed for reproducible shuffling |
