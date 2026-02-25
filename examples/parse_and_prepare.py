#!/usr/bin/env python
"""
Parse, inspect, and prepare structural biology datasets.

Computational scientists often spend 60%+ of their time wrangling PDB/CIF/SDF
files. This example shows how Molfun's unified parsers handle the pain points:

  - Parse PDB, mmCIF, SDF, MOL2, A3M, FASTA with one API
  - Auto-detect format by extension
  - Extract coordinates, sequences, bonds as clean NumPy arrays
  - Build a merged protein-ligand dataset ready for training

Requires: pip install molfun
"""

from pathlib import Path

import numpy as np
import torch

from molfun.data.parsers import auto_parser
from molfun.data.parsers.pdb import PDBParser
from molfun.data.parsers.mmcif import CIFParser
from molfun.data.parsers.sdf import SDFParser
from molfun.data.parsers.mol2 import MOL2Parser
from molfun.data.parsers.a3m import A3MParser
from molfun.data.parsers.fasta import FASTAParser
from molfun.data.sources.pdb import PDBFetcher

# ── 1. Parse a protein structure ─────────────────────────────────────

print("=== PDB Parsing ===\n")

fetcher = PDBFetcher(cache_dir="data/structures", fmt="pdb")
paths = fetcher.fetch(["1a2b"])

if paths:
    parser = auto_parser(paths[0])  # auto-detects PDB by extension
    result = parser.parse(paths[0])

    print(f"PDB: {paths[0]}")
    print(f"  Chains:     {result.chains}")
    print(f"  Residues:   {len(result.sequence)}")
    print(f"  Sequence:   {result.sequence[:50]}...")
    print(f"  Coords:     shape {result.coords.shape}  (N_res × 3)")
    print(f"  B-factors:  mean={result.b_factors.mean():.1f}")
    print()

# ── 2. Parse mmCIF (modern PDB format) ──────────────────────────────

print("=== mmCIF Parsing ===\n")

fetcher_cif = PDBFetcher(cache_dir="data/structures", fmt="cif")
paths_cif = fetcher_cif.fetch(["7bv2"])

if paths_cif:
    cif = CIFParser()
    result = cif.parse(paths_cif[0])
    print(f"mmCIF: {paths_cif[0]}")
    print(f"  Residues: {len(result.sequence)}")
    print(f"  Coords:   {result.coords.shape}")
    print()

# ── 3. Parse ligands (SDF / MOL2) ───────────────────────────────────

print("=== Ligand Parsing ===\n")

# Example: parse an SDF file from a docking run
sdf_example = Path("data/ligands/docked_poses.sdf")
if sdf_example.exists():
    sdf = SDFParser()
    molecules = sdf.parse(str(sdf_example))

    for mol in molecules:
        print(f"  Molecule: {mol.name}")
        print(f"    Atoms: {len(mol.atoms)}")
        print(f"    Bonds: {len(mol.bonds)}")
        print(f"    Properties: {mol.properties}")
else:
    print("  (No SDF file found — download ligands to data/ligands/)")
    print("  Example: from ChEMBL or ZINC database")
    print()

# You can also create SDF content programmatically for testing:
sdf_content = """\

     RDKit          3D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5400    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.3100    1.3300    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  2  0
M  END
$$$$
"""
tmp = Path("/tmp/demo_ligand.sdf")
tmp.write_text(sdf_content)
sdf = SDFParser()
mols = sdf.parse(str(tmp))
print(f"  Parsed SDF: {len(mols)} molecule(s), {len(mols[0].atoms)} atoms, {len(mols[0].bonds)} bonds")
print()

# ── 4. Parse MSA (A3M format) ────────────────────────────────────────

print("=== A3M (MSA) Parsing ===\n")

a3m_content = ">query\nMKFLIVALIAGGS\n>hit1\nMKFL-VALIDGGS\n>hit2\nMKaFLIVALIAGGS\n"
tmp_a3m = Path("/tmp/demo.a3m")
tmp_a3m.write_text(a3m_content)

a3m = A3MParser(max_seq_len=256)
alignment = a3m.parse(str(tmp_a3m))

print(f"  Sequences: {alignment.sequences.shape[0]}")
print(f"  Length:    {alignment.sequences.shape[1]}")
print(f"  Deletions: {alignment.deletion_matrix.sum().item():.0f} total")
print(f"  Query seq: {a3m_content.split(chr(10))[1]}")
print()

# ── 5. Parse FASTA ───────────────────────────────────────────────────

print("=== FASTA Parsing ===\n")

fasta_content = ">sp|P0DTD1|R1AB_SARS2 SARS-CoV-2 Replicase\nMESLVPGFNEKTH\n>sp|P0DTC2|SPIKE_SARS2\nMFVFLVLLPLVSSQ\n"
tmp_fasta = Path("/tmp/demo.fasta")
tmp_fasta.write_text(fasta_content)

fasta = FASTAParser()
parsed = fasta.parse(str(tmp_fasta))
print(f"  Sequences: {parsed.sequences.shape[0]}")
for i in range(parsed.sequences.shape[0]):
    print(f"  [{i}] length={parsed.sequences[i].sum().item():.0f} (non-gap positions)")
print()

# ── 6. Build a protein-ligand dataset ────────────────────────────────

print("=== Building a merged dataset ===\n")

# Typical workflow for structure-based drug discovery:
# 1. Parse protein structures
# 2. Parse ligand conformers
# 3. Combine with affinity labels
# 4. Create a PyTorch dataset

# Here we show the pattern:
print("  Workflow for protein-ligand affinity prediction:")
print("    1. Fetch PDB structures  →  PDBFetcher + PDBParser/CIFParser")
print("    2. Extract ligands       →  SDFParser (from docking or ChEMBL)")
print("    3. Align MSAs            →  A3MParser (from ColabFold)")
print("    4. Load labels           →  AffinitySource (PDBbind CSV)")
print("    5. Create StructureDataset + DataLoader")
print("    6. Fine-tune with molfun.training strategies")
print()
print("  See finetune_affinity.py for the complete training loop.")

# ── 7. Auto-detect and batch parse ───────────────────────────────────

print("=== Auto-detection ===\n")

from molfun.data.parsers import PARSER_REGISTRY

print(f"  Registered formats: {list(PARSER_REGISTRY.keys())}")
print()

# auto_parser picks the right class by extension:
for path_str, desc in [
    ("/tmp/demo.a3m", "alignment"),
    ("/tmp/demo.fasta", "sequences"),
    ("/tmp/demo_ligand.sdf", "molecule"),
]:
    p = auto_parser(path_str)
    print(f"  {path_str} → {type(p).__name__} ({desc})")

print("\nDone. All parsed data is available as NumPy arrays / torch tensors.")
