# Molfun Examples

Practical examples for computational scientists working with molecular ML models.
Each script is self-contained — read the docstring at the top for requirements and context.

## Quick Start

```bash
pip install molfun[openfold]  # core + OpenFold support
```

## Examples

| Script | What it does | Time | GPU? |
|--------|-------------|------|------|
| [`finetune_affinity.py`](finetune_affinity.py) | Full pipeline: fetch PDBs → build dataset → LoRA fine-tune → evaluate → push to Hub | ~15 min | Yes |
| [`custom_architecture.py`](custom_architecture.py) | Build custom models by mixing attention, blocks, structure modules | ~1 min | No |
| [`agent_architecture_search.py`](agent_architecture_search.py) | LLM agent autonomously searches architecture space | hours | Yes |
| [`parse_and_prepare.py`](parse_and_prepare.py) | Parse PDB/CIF/SDF/MOL2/A3M/FASTA files with one unified API | ~1 min | No |
| [`tracked_experiment.py`](tracked_experiment.py) | Log to W&B + Comet + MLflow + HF Hub simultaneously | ~1 min | No |
| [`streaming_cloud.py`](streaming_cloud.py) | Stream structures from S3/MinIO/GCS without downloading | ~1 min | No |

## By Use Case

**I want to fine-tune a model on my data:**
→ Start with `finetune_affinity.py`, replace the PDB IDs and labels with yours.

**I want to try a new architecture idea:**
→ `custom_architecture.py` shows how to swap attention, blocks, and structure modules in 5 lines.

**I want to automate hyperparameter / architecture search:**
→ `agent_architecture_search.py` sets up an LLM agent that runs experiments while you sleep.

**I have PDB/CIF/SDF files and need to parse them:**
→ `parse_and_prepare.py` covers every format Molfun supports with working examples.

**I want my experiments tracked properly:**
→ `tracked_experiment.py` shows multi-backend logging in one config block.

**My dataset lives in S3 / MinIO / GCS:**
→ `streaming_cloud.py` streams structures without downloading the full dataset.

## Common Patterns

### Install only what you need

```bash
pip install molfun                      # core only
pip install molfun[openfold]            # + OpenFold support
pip install molfun[wandb,comet]         # + experiment tracking
pip install molfun[streaming]           # + cloud streaming (fsspec)
pip install molfun[agents]              # + LLM agent framework
pip install molfun[hub]                 # + Hugging Face Hub
pip install molfun[dev]                 # everything for development
```

### Run an example

```bash
cd examples/
python parse_and_prepare.py             # no GPU needed
python custom_architecture.py           # no GPU needed
python finetune_affinity.py             # needs GPU + OpenFold weights
```

### Use the CLI instead

```bash
molfun fetch-pdb 1a2b 7bv2 --fmt cif
molfun parse data/structures/1a2b.cif --format json
molfun finetune --model openfold --strategy lora --data-dir data/
molfun registry --category attention
```
