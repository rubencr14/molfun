# Parsers

File parsers for common structural biology and cheminformatics formats.
Each parser reads a file and returns a structured Python object.

## Quick Start

```python
from molfun.data.parsers import PDBParser, CIFParser, A3MParser

# Parse a PDB file
parser = PDBParser()
structure = parser.parse("protein.pdb")

# Parse an mmCIF file
parser = CIFParser()
structure = parser.parse("protein.cif")

# Parse an MSA
parser = A3MParser()
msa = parser.parse("alignment.a3m")
```

## PDBParser

::: molfun.data.parsers.pdb.PDBParser
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Parse PDB format files into structured data.

```python
from molfun.data.parsers import PDBParser

parser = PDBParser()
structure = parser.parse("protein.pdb")

print(structure.sequence)        # "MKFLILLFNILCLFPVLAADNH..."
print(structure.positions.shape) # (N_residues, 37, 3)
print(structure.resolution)      # 2.1
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_hetatm` | `bool` | `False` | Include HETATM records |
| `model_idx` | `int` | `0` | NMR model index to use |

---

## CIFParser

::: molfun.data.parsers.mmcif.CIFParser
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Parse mmCIF format files (PDBx/mmCIF).

```python
from molfun.data.parsers import CIFParser

parser = CIFParser()
structure = parser.parse("protein.cif")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_hetatm` | `bool` | `False` | Include non-polymer entities |
| `auth_chains` | `bool` | `True` | Use author chain IDs (vs label chain IDs) |

---

## A3MParser

::: molfun.data.parsers.a3m.A3MParser
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Parse A3M format multiple sequence alignments.

```python
from molfun.data.parsers import A3MParser

parser = A3MParser()
msa = parser.parse("alignment.a3m")

print(msa.sequences)    # list of aligned sequences
print(msa.descriptions) # list of sequence headers
print(len(msa))         # number of sequences
```

---

## FASTAParser

::: molfun.data.parsers.fasta.FASTAParser
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Parse FASTA format sequence files.

```python
from molfun.data.parsers import FASTAParser

parser = FASTAParser()
entries = parser.parse("sequences.fasta")

for entry in entries:
    print(entry.header, entry.sequence)
```

---

## SDFParser

::: molfun.data.parsers.sdf.SDFParser
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Parse SDF (Structure-Data File) format for small molecules.

```python
from molfun.data.parsers import SDFParser

parser = SDFParser()
molecules = parser.parse("ligands.sdf")

for mol in molecules:
    print(mol.name, mol.num_atoms, mol.coordinates.shape)
```

---

## Mol2Parser

::: molfun.data.parsers.mol2.MOL2Parser
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Parse Tripos Mol2 format files.

```python
from molfun.data.parsers import Mol2Parser

parser = Mol2Parser()
molecules = parser.parse("ligand.mol2")
```

---

## Summary

| Parser | Format | Use Case |
|--------|--------|----------|
| `PDBParser` | `.pdb` | Legacy protein structures |
| `CIFParser` | `.cif` | Modern protein structures (recommended) |
| `A3MParser` | `.a3m` | Multiple sequence alignments |
| `FASTAParser` | `.fasta` / `.fa` | Protein/nucleotide sequences |
| `SDFParser` | `.sdf` | Small molecule structures |
| `Mol2Parser` | `.mol2` | Small molecules with atom types |
