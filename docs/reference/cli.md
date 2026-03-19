# CLI Reference

Molfun provides a command-line interface built with [Typer](https://typer.tiangolo.com/).
The entry point is `molfun`.

```bash
molfun --help
```

## Commands Overview

| Command | Description |
|---------|-------------|
| `molfun structure` | Predict protein structure from sequence |
| `molfun affinity` | Predict protein-ligand binding affinity |
| `molfun info` | Display model and environment information |
| `molfun fetch-pdb` | Download PDB structures |
| `molfun fetch-msa` | Generate/fetch MSAs for sequences |
| `molfun fetch-domain` | Fetch domain annotations |
| `molfun parse` | Parse structure/sequence files |
| `molfun registry` | List registered modules |
| `molfun agent` | Run an AI agent for automated workflows |
| `molfun eval` | Evaluate a model on a benchmark |
| `molfun benchmark` | Run performance benchmarks |
| `molfun run` | Run a training/export script |
| `molfun push` | Push a model to the Hub |
| `molfun pull` | Pull a model from the Hub |
| `molfun push-dataset` | Push a dataset to the Hub |

---

## structure

Predict the 3D structure of a protein.

```bash
molfun structure \
    --sequence "MKFLILLFNILCLFPVLAADNH..." \
    --model openfold_v2 \
    --output prediction.pdb \
    --device cuda \
    --num-recycles 3
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--sequence` / `-s` | `str` | *required* | Amino acid sequence |
| `--fasta` / `-f` | `Path` | -- | FASTA file with sequence(s) |
| `--model` / `-m` | `str` | `openfold_v2` | Pretrained model name |
| `--output` / `-o` | `Path` | `prediction.pdb` | Output PDB file |
| `--device` | `str` | `cpu` | Compute device |
| `--num-recycles` | `int` | `3` | Recycling iterations |
| `--msa` | `Path` | -- | Precomputed A3M MSA file |

---

## affinity

Predict binding affinity for a protein-ligand pair.

```bash
molfun affinity \
    --sequence "MKFLILLFNILCLFPVLAADNH..." \
    --ligand ligand.sdf \
    --model openfold_v2
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--sequence` / `-s` | `str` | *required* | Protein sequence |
| `--ligand` / `-l` | `Path` | *required* | Ligand file (SDF) |
| `--model` / `-m` | `str` | `openfold_v2` | Model name |
| `--device` | `str` | `cpu` | Compute device |

---

## info

Display information about the Molfun installation and available models.

```bash
molfun info
molfun info --model openfold_v2
```

| Flag | Type | Description |
|------|------|-------------|
| `--model` / `-m` | `str` | Show details for a specific model |
| `--verbose` / `-v` | `bool` | Show extended system information |

---

## fetch-pdb

Download PDB structures from RCSB.

```bash
# By IDs
molfun fetch-pdb --ids 1ABC 2DEF 3GHI --output-dir ./pdb_files

# By query
molfun fetch-pdb \
    --resolution-max 2.5 \
    --organism "Homo sapiens" \
    --max-results 100 \
    --output-dir ./pdb_files
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--ids` | `list[str]` | -- | PDB IDs to download |
| `--resolution-max` | `float` | -- | Max resolution filter |
| `--organism` | `str` | -- | Source organism filter |
| `--method` | `str` | -- | Experimental method filter |
| `--max-results` | `int` | -- | Maximum entries to download |
| `--output-dir` / `-o` | `Path` | `./pdb` | Output directory |
| `--format` | `str` | `mmcif` | File format (`pdb` or `mmcif`) |

---

## fetch-msa

Generate or fetch MSAs for protein sequences.

```bash
molfun fetch-msa \
    --sequence "MKFLILLFNILCLFPVLAADNH..." \
    --output alignment.a3m \
    --backend colabfold
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--sequence` / `-s` | `str` | -- | Amino acid sequence |
| `--fasta` / `-f` | `Path` | -- | FASTA file with sequence(s) |
| `--output` / `-o` | `Path` | `alignment.a3m` | Output A3M file |
| `--backend` | `str` | `colabfold` | MSA backend |

---

## fetch-domain

Fetch domain annotations for a protein.

```bash
molfun fetch-domain --uniprot P12345
```

| Flag | Type | Description |
|------|------|-------------|
| `--uniprot` / `-u` | `str` | UniProt accession |
| `--pdb` | `str` | PDB ID |

---

## parse

Parse structure and sequence files.

```bash
molfun parse protein.pdb --format pdb
molfun parse alignment.a3m --format a3m --stats
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `FILE` | `Path` | *required* | Input file (positional) |
| `--format` | `str` | auto | File format (auto-detected from extension) |
| `--stats` | `bool` | `False` | Print summary statistics |
| `--output` / `-o` | `Path` | -- | Output file for converted data |

---

## registry

List registered modular components.

```bash
molfun registry                   # list all registries
molfun registry attention         # list attention modules
molfun registry --details block   # detailed info per block
```

| Flag | Type | Description |
|------|------|-------------|
| `TYPE` | `str` | Optional: registry type to list (`attention`, `block`, `embedder`, `structure_module`, `loss`) |
| `--details` / `-d` | `bool` | Show detailed information for each entry |

---

## agent

Run an AI agent for automated experiment workflows.

```bash
molfun agent "Fine-tune openfold on CASP15 with LoRA rank 8"
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `PROMPT` | `str` | *required* | Natural language instruction (positional) |
| `--model` | `str` | `gpt-4` | LLM model to use |
| `--dry-run` | `bool` | `False` | Print plan without executing |

---

## eval

Evaluate a model on a benchmark dataset.

```bash
molfun eval \
    --model openfold_v2 \
    --benchmark casp15 \
    --device cuda \
    --output results.json
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` / `-m` | `str` | *required* | Model name or path |
| `--benchmark` / `-b` | `str` | *required* | Benchmark name (e.g., `casp15`, `cameo`) |
| `--device` | `str` | `cpu` | Compute device |
| `--output` / `-o` | `Path` | `results.json` | Output file |

---

## benchmark

Run performance benchmarks (throughput, latency, memory).

```bash
molfun benchmark \
    --model openfold_v2 \
    --sequence-lengths 100 200 500 \
    --device cuda \
    --warmup 5 \
    --iterations 20
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` / `-m` | `str` | *required* | Model name |
| `--sequence-lengths` | `list[int]` | `[100, 200, 500]` | Sequence lengths to benchmark |
| `--device` | `str` | `cpu` | Compute device |
| `--warmup` | `int` | `5` | Warmup iterations |
| `--iterations` | `int` | `20` | Benchmark iterations |

---

## run

Run a training or export script.

```bash
molfun run train.py --gpus 4
molfun run export --format onnx --model openfold_v2 --output model.onnx
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `SCRIPT` | `str` | *required* | Script path or subcommand |
| `--gpus` | `int` | `1` | Number of GPUs (launches with torchrun for > 1) |

---

## push / pull

Push and pull models from the Molfun Hub.

```bash
# Push
molfun push ./my_model --repo myorg/finetuned-openfold --private

# Pull
molfun pull myorg/finetuned-openfold --output ./my_model
```

### push

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `PATH` | `Path` | *required* | Local model directory |
| `--repo` / `-r` | `str` | *required* | Hub repository ID |
| `--private` | `bool` | `False` | Make repository private |
| `--message` | `str` | -- | Commit message |

### pull

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `REPO` | `str` | *required* | Hub repository ID |
| `--output` / `-o` | `Path` | `./` | Output directory |
| `--revision` | `str` | -- | Specific revision/tag |

---

## push-dataset

Push a dataset to the Hub.

```bash
molfun push-dataset ./my_dataset --repo myorg/structure-dataset --format parquet
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `PATH` | `Path` | *required* | Local dataset directory |
| `--repo` / `-r` | `str` | *required* | Hub repository ID |
| `--format` | `str` | `parquet` | Dataset format |
| `--private` | `bool` | `False` | Make repository private |
