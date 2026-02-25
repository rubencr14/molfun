"""
Molfun CLI — entry point.

Commands
--------
    molfun structure    Fine-tune on PDB structures (FAPE loss, no labels)
    molfun affinity     Fine-tune on binding affinity data (ΔG regression)
    molfun info         Show system info, available strategies / losses / heads

    molfun fetch-pdb    Download PDB structures from RCSB
    molfun fetch-msa    Fetch MSAs via ColabFold API or load precomputed
    molfun parse        Inspect / validate biological data files
    molfun registry     List registered modules and components
    molfun agent        Launch autonomous research agent
    molfun eval         Evaluate a trained checkpoint on test data
    molfun benchmark    Run performance benchmarks

    molfun push         Push model checkpoint to Hugging Face Hub
    molfun pull         Download model from Hugging Face Hub
    molfun push-dataset Push dataset to Hugging Face Hub
"""

import typer

from molfun.cli.finetune import structure, affinity
from molfun.cli.info import info
from molfun.cli.fetch import fetch_pdb, fetch_msa
from molfun.cli.parse import parse
from molfun.cli.registry import registry
from molfun.cli.agent import agent
from molfun.cli.eval import eval_model
from molfun.cli.benchmark import benchmark
from molfun.cli.hub import push, pull, push_dataset

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

# Fine-tuning
app.command("structure")(structure)
app.command("affinity")(affinity)

# Data
app.command("fetch-pdb")(fetch_pdb)
app.command("fetch-msa")(fetch_msa)
app.command("parse")(parse)

# Inspection
app.command("info")(info)
app.command("registry")(registry)

# Agent
app.command("agent")(agent)

# Evaluation & Benchmarks
app.command("eval")(eval_model)
app.command("benchmark")(benchmark)

# Hugging Face Hub
app.command("push")(push)
app.command("pull")(pull)
app.command("push-dataset")(push_dataset)
