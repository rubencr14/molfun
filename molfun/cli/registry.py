"""
CLI command for listing registered modules and components.
"""

from __future__ import annotations
from typing import Annotated, Optional
from enum import Enum

import typer


class RegistryType(str, Enum):
    all = "all"
    attention = "attention"
    blocks = "blocks"
    structure = "structure"
    embedders = "embedders"
    strategies = "strategies"
    heads = "heads"
    losses = "losses"
    parsers = "parsers"


def registry(
    category: Annotated[RegistryType, typer.Argument(help="Registry to list.")] = RegistryType.all,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show extra details.")] = False,
):
    """List all registered modules, strategies, losses, and parsers."""
    sections = _gather_sections(category)

    for name, items in sections.items():
        typer.echo(f"\n{name}")
        typer.echo("─" * len(name))
        if isinstance(items, dict):
            for key, val in items.items():
                typer.echo(f"  {key}: {val}" if verbose else f"  {key}")
        elif isinstance(items, list):
            for item in items:
                typer.echo(f"  {item}")
        else:
            typer.echo(f"  {items}")

    typer.echo("")


def _gather_sections(category: RegistryType) -> dict:
    sections = {}

    if category in (RegistryType.all, RegistryType.attention):
        try:
            from molfun.modules import ATTENTION_REGISTRY
            sections["Attention mechanisms"] = {
                name: ATTENTION_REGISTRY.get(name).__doc__.split("\n")[0].strip()
                if ATTENTION_REGISTRY.get(name).__doc__ else ""
                for name in ATTENTION_REGISTRY.list()
            }
        except Exception:
            sections["Attention mechanisms"] = ["(not available)"]

    if category in (RegistryType.all, RegistryType.blocks):
        try:
            from molfun.modules import BLOCK_REGISTRY
            sections["Trunk blocks"] = {
                name: BLOCK_REGISTRY.get(name).__doc__.split("\n")[0].strip()
                if BLOCK_REGISTRY.get(name).__doc__ else ""
                for name in BLOCK_REGISTRY.list()
            }
        except Exception:
            sections["Trunk blocks"] = ["(not available)"]

    if category in (RegistryType.all, RegistryType.structure):
        try:
            from molfun.modules import STRUCTURE_MODULE_REGISTRY
            sections["Structure modules"] = {
                name: STRUCTURE_MODULE_REGISTRY.get(name).__doc__.split("\n")[0].strip()
                if STRUCTURE_MODULE_REGISTRY.get(name).__doc__ else ""
                for name in STRUCTURE_MODULE_REGISTRY.list()
            }
        except Exception:
            sections["Structure modules"] = ["(not available)"]

    if category in (RegistryType.all, RegistryType.embedders):
        try:
            from molfun.modules import EMBEDDER_REGISTRY
            sections["Embedders"] = {
                name: EMBEDDER_REGISTRY.get(name).__doc__.split("\n")[0].strip()
                if EMBEDDER_REGISTRY.get(name).__doc__ else ""
                for name in EMBEDDER_REGISTRY.list()
            }
        except Exception:
            sections["Embedders"] = ["(not available)"]

    if category in (RegistryType.all, RegistryType.strategies):
        sections["Training strategies"] = [
            "head_only  — Freeze trunk, train only the head",
            "lora       — LoRA adapters on trunk + head",
            "partial    — Unfreeze last N blocks + structure module",
            "full       — Full fine-tuning with layer-wise LR decay",
        ]

    if category in (RegistryType.all, RegistryType.heads):
        try:
            from molfun.models.structure import HEAD_REGISTRY
            sections["Prediction heads"] = list(HEAD_REGISTRY.keys())
        except Exception:
            sections["Prediction heads"] = ["affinity", "structure"]

    if category in (RegistryType.all, RegistryType.losses):
        try:
            from molfun.losses import LOSS_REGISTRY
            sections["Loss functions"] = list(LOSS_REGISTRY.keys())
        except Exception:
            sections["Loss functions"] = ["(not available)"]

    if category in (RegistryType.all, RegistryType.parsers):
        from molfun.data.parsers import PARSER_REGISTRY
        sections["File parsers"] = {
            ext: cls.__name__ for ext, cls in sorted(PARSER_REGISTRY.items())
        }

    return sections
