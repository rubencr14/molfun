"""
Molfun CLI — entry point.

Commands
--------
    molfun structure    Fine-tune on PDB structures (FAPE loss, no labels)
    molfun affinity     Fine-tune on binding affinity data (ΔG regression)
    molfun info         Show system info, available strategies / losses / heads
"""

import typer

from molfun.cli.finetune import structure, affinity
from molfun.cli.info import info

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
app.command("structure")(structure)
app.command("affinity")(affinity)
app.command("info")(info)
