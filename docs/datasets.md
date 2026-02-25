# Datasets for Serious Fine-Tuning with Molfun

This guide lists the main public datasets for training, fine-tuning, and evaluating molecular ML models with Molfun, including download instructions and how each one connects to the library.

---

## Protein–Ligand Binding Affinity

The primary use case for Molfun fine-tuning: predict pKd/pKi/IC50 from 3D structure.

| Dataset | Size | Contents | Download | Molfun integration |
|---------|------|----------|----------|-------------------|
| **PDBbind v2020** | ~19K complexes (refined: 5,316) | 3D structures + experimental pKd/pKi/IC50 | [pdbbind.org.cn](http://www.pdbbind.org.cn) (free registration) | `AffinitySource.from_pdbbind_index()` + `PDBFetcher` |
| **BindingDB** | ~2.7M measurements | Experimental affinity, some with structure | [bindingdb.org](https://www.bindingdb.org/rwd/bind/index.jsp) → bulk TSV download | Parse TSV → `AffinitySource.from_csv()` |
| **ChEMBL 34** | ~2.4M compounds, 20M+ activities | Bioactivity, SMILES, targets | [ebi.ac.uk/chembl](https://www.ebi.ac.uk/chembl/) → FTP dump or API | SDF via `SDFParser`, labels via CSV |
| **ATOM3D LBA** | Pre-packaged task | Ligand binding affinity with controlled splits | [atom3d.ai](https://www.atom3d.ai) → `pip install atom3d` | `BenchmarkSuite.atom3d_lba()` |

### Quick start: PDBbind

```bash
# 1. Register at pdbbind.org.cn (free academic license)
# 2. Download PDBbind_v2020_refined.tar.gz
# 3. Extract — includes structures + INDEX_refined_data.2020

tar xzf PDBbind_v2020_refined.tar.gz -C data/pdbbind/
```

```python
from molfun.data.sources.affinity import AffinitySource

labels = AffinitySource.from_pdbbind_index("data/pdbbind/INDEX_refined_data.2020")
# Returns dict: {pdb_id: pKd_value}
```

---

## Structure Prediction / Folding

For training or evaluating structure prediction models.

| Dataset | Size | Contents | Download |
|---------|------|----------|----------|
| **Full PDB** | ~215K structures (~50 GB CIF) | All experimental structures deposited | `rsync -rlpt rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ data/pdb/` |
| **AlphaFold DB** | ~214M predicted structures | Full proteomes predicted by AF2 | [alphafold.ebi.ac.uk/download](https://alphafold.ebi.ac.uk/download) → bulk HTTPS/FTP by organism |
| **CASP15 targets** | ~100 targets | Evaluation targets with ground truth | [predictioncenter.org/casp15](https://predictioncenter.org/casp15/targetlist.cgi) |
| **CAMEO** | Weekly | New targets every week for continuous evaluation | [cameo3d.org](https://www.cameo3d.org) |

### Quick start: fetch structures

```python
from molfun.data.sources.pdb import PDBFetcher

fetcher = PDBFetcher(cache_dir="data/structures", fmt="cif")
paths = fetcher.fetch(["7bv2", "3hb5", "4yne", "1a2b"])
```

### Quick start: bulk PDB mirror

```bash
# Full PDB mirror via rsync (only downloads new/changed files on re-run)
rsync -rlpt --info=progress2 \
  rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ \
  data/pdb_mirror/
```

---

## MSAs (Multiple Sequence Alignments)

Required by AlphaFold-family models for co-evolutionary features.

| Source | Size | Format | Notes |
|--------|------|--------|-------|
| **OpenProteinSet** | ~600 GB | Precomputed A3M | MSAs used to train OpenFold; **S3 public bucket** |
| **UniRef30** (ColabFold) | ~70 GB | A3M/HHM | For running ColabFold MMseqs2 locally |
| **BFD** (Big Fantastic Database) | ~1.8 TB | A3M/HHM | Used by original AlphaFold 2 training |

### Quick start: OpenProteinSet (recommended)

Precomputed MSAs for ~140K PDB chains, available on public S3 — no download needed with Molfun's streaming:

```python
from molfun.data.datasets.streaming import StreamingStructureDataset

# Stream directly from public S3 (no AWS credentials needed)
import os
os.environ["AWS_NO_SIGN_REQUEST"] = "1"

dataset = StreamingStructureDataset(
    root="s3://openfold/alignment_dbs/",
    glob_pattern="*.a3m",
    max_seq_len=512,
)
```

### Quick start: precomputed MSAs (local)

```python
from molfun.data.sources.msa import MSAProvider

# Option 1: single-sequence (no MSA, fast, for prototyping)
msa = MSAProvider("single")

# Option 2: precomputed A3M directory
msa = MSAProvider("precomputed", msa_dir="data/msas/")

# Option 3: ColabFold API (online, rate-limited)
msa = MSAProvider("colabfold")
```

---

## Protein Fitness / Function

For predicting the effect of mutations on protein function.

| Dataset | Size | Contents | Download |
|---------|------|----------|----------|
| **FLIP** | 5 tasks | Fitness landscapes (AAV, GB1, meltome, thermostability) | [github.com/J-SNACKKB/FLIP](https://github.com/J-SNACKKB/FLIP) |
| **ProteinGym** | 87 DMS datasets, >2.5M variants | Deep mutational scanning across diverse proteins | [proteingym.org](https://proteingym.org) |
| **TAPE** | 5 tasks | Stability, fluorescence, remote homology, contact prediction | [github.com/songlab-cal/tape](https://github.com/songlab-cal/tape) |

```python
# FLIP can be evaluated directly with Molfun's benchmark suite:
from molfun.benchmarks import BenchmarkSuite

suite = BenchmarkSuite.flip(data_dir="data/flip")
# Evaluates on AAV, GB1, and meltome fitness landscapes
```

---

## Docking / Drug Design

For structure-based drug discovery and pose prediction.

| Dataset | Size | Contents | Download |
|---------|------|----------|----------|
| **CrossDocked2020** | ~22K cross-docked poses | Protein–ligand poses for pose prediction | [Zenodo via gnina/models](https://github.com/gnina/models) |
| **ZINC20** | 1.4 billion molecules | 3D conformers, SDF format | [zinc20.docking.org](https://zinc20.docking.org) |
| **PoseBusters** | ~428 complexes | Benchmark for generated pose quality | [github.com/maabuu/posebusters](https://github.com/maabuu/posebusters) |

Ligand files can be parsed with Molfun's unified parser system:

```python
from molfun.data.parsers import auto_parser

parser = auto_parser("docked_poses.sdf")
molecules = parser.parse_file("docked_poses.sdf")

for mol in molecules:
    print(f"{mol.name}: {len(mol.atoms)} atoms, {len(mol.bonds)} bonds")
```

---

## Recommended Setup for a Serious Training Run

The minimal combination for a publishable affinity prediction paper:

```bash
# 1. PDBbind refined set (standard affinity benchmark)
#    Download from pdbbind.org.cn → extract to data/pdbbind/

# 2. Precomputed MSAs from OpenProteinSet (free, public S3)
#    No download needed — stream directly with Molfun

# 3. ATOM3D for standardized evaluation
pip install atom3d
python -c "import atom3d; atom3d.datasets.download_dataset('LBA', 'data/atom3d/LBA')"
```

```python
from molfun.data.sources.pdb import PDBFetcher
from molfun.data.sources.affinity import AffinitySource
from molfun.data.sources.msa import MSAProvider
from molfun.data.datasets.structure import StructureDataset
from molfun.data.splits import DataSplitter
from molfun.models.structure import MolfunStructureModel
from molfun.training import LoRAFinetune
from molfun.benchmarks import BenchmarkSuite, ModelEvaluator

# Load data
labels = AffinitySource.from_pdbbind_index("data/pdbbind/INDEX_refined_data.2020")
fetcher = PDBFetcher(cache_dir="data/pdbbind/structures", fmt="cif")
paths = fetcher.fetch(list(labels.keys()))
msa = MSAProvider("precomputed", msa_dir="data/msas/")

# Build dataset + identity-based split (avoids homology leakage)
dataset = StructureDataset(pdb_paths=paths, labels=labels, msa_provider=msa)
train, val, test = DataSplitter.by_sequence_identity(dataset, threshold=0.3)

# Fine-tune
model = MolfunStructureModel("openfold", config=config, weights="ckpt.pt",
                              device="cuda", head="affinity",
                              head_config={"single_dim": 384, "hidden_dim": 128})
strategy = LoRAFinetune(rank=8, lr_lora=2e-4, lr_head=1e-3)
strategy.fit(model, train_loader, val_loader, epochs=20)

# Evaluate on standard benchmarks
suite = BenchmarkSuite.pdbbind()
report = ModelEvaluator(model, suite, device="cuda").run()
print(report.to_markdown())
print(report.to_latex())  # copy into your paper
```

---

## Data Format Compatibility

All datasets above produce files in formats that Molfun parsers handle natively:

| Format | Parser | Typical source |
|--------|--------|---------------|
| `.pdb` | `PDBParser` | PDB, PDBbind, CrossDocked |
| `.cif` | `CIFParser` | PDB (modern), AlphaFold DB |
| `.sdf` | `SDFParser` | ChEMBL, ZINC, docking outputs |
| `.mol2` | `MOL2Parser` | Docking tools (GOLD, Surflex) |
| `.a3m` | `A3MParser` | OpenProteinSet, ColabFold, HHblits |
| `.fasta` | `FASTAParser` | UniProt, sequence databases |

Auto-detection by extension:

```python
from molfun.data.parsers import auto_parser

parser = auto_parser("structure.cif")  # → CIFParser
parser = auto_parser("alignment.a3m")  # → A3MParser
parser = auto_parser("ligand.sdf")     # → SDFParser
```
