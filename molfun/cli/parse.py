"""
CLI command for inspecting and validating biological data files.
"""

from __future__ import annotations
from pathlib import Path
from typing import Annotated, Optional

import typer


def parse(
    paths: Annotated[list[Path], typer.Argument(help="Files to parse (PDB, CIF, SDF, MOL2, A3M, FASTA).")],
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
    max_seq_len: Annotated[int, typer.Option(help="Max sequence length for structure parsers.")] = 9999,
):
    """Parse and inspect biological data files (structures, ligands, alignments)."""
    import json

    from molfun.data.parsers import auto_parser
    from molfun.data.parsers.base import (
        BaseStructureParser,
        BaseLigandParser,
        BaseAlignmentParser,
    )

    results = []

    for path in paths:
        if not path.exists():
            typer.echo(f"File not found: {path}", err=True)
            continue

        try:
            parser = auto_parser(str(path), max_seq_len=max_seq_len)
        except ValueError as e:
            typer.echo(f"Unsupported format: {path} — {e}", err=True)
            continue

        try:
            if isinstance(parser, BaseStructureParser):
                result = parser.parse_file(str(path))
                info = _structure_info(path, result)
            elif isinstance(parser, BaseLigandParser):
                mols = parser.parse_file(str(path))
                info = _ligand_info(path, mols)
            elif isinstance(parser, BaseAlignmentParser):
                result = parser.parse_file(str(path))
                info = _alignment_info(path, result)
            else:
                info = {"file": str(path), "error": "Unknown parser type"}
        except Exception as e:
            info = {"file": str(path), "error": str(e)}

        results.append(info)

    if json_output:
        typer.echo(json.dumps(results, indent=2, default=str))
    else:
        for info in results:
            _print_info(info)


def _structure_info(path: Path, parsed) -> dict:
    seq = parsed.sequence
    return {
        "file": str(path),
        "type": "structure",
        "residues": parsed.seq_length.item(),
        "chains": parsed.chain_ids or ["?"],
        "sequence_preview": seq[:60] + ("..." if len(seq) > 60 else ""),
        "ca_coverage": f"{parsed.all_atom_mask.sum().item():.0f}/{parsed.seq_length.item()}",
        "has_gaps": "X" in seq,
    }


def _ligand_info(path: Path, molecules: list) -> dict:
    mols_info = []
    for mol in molecules[:20]:
        mols_info.append({
            "name": mol.name,
            "atoms": mol.num_atoms,
            "bonds": mol.num_bonds,
            "elements": sorted(set(mol.elements)),
            "properties": list(mol.properties.keys())[:5] if mol.properties else [],
        })
    return {
        "file": str(path),
        "type": "ligand",
        "num_molecules": len(molecules),
        "molecules": mols_info,
    }


def _alignment_info(path: Path, parsed) -> dict:
    return {
        "file": str(path),
        "type": "alignment",
        "depth": parsed.depth,
        "query_length": parsed.length,
        "headers_preview": parsed.headers[:5] if parsed.headers else [],
    }


def _print_info(info: dict) -> None:
    typer.echo(f"\n{'─' * 60}")
    typer.echo(f"  File: {info['file']}")

    if "error" in info:
        typer.echo(f"  ERROR: {info['error']}")
        return

    ftype = info.get("type", "unknown")
    typer.echo(f"  Type: {ftype}")

    if ftype == "structure":
        typer.echo(f"  Residues: {info['residues']}")
        typer.echo(f"  Chains: {', '.join(info['chains'])}")
        typer.echo(f"  CA coverage: {info['ca_coverage']}")
        typer.echo(f"  Has unknown (X): {info['has_gaps']}")
        typer.echo(f"  Sequence: {info['sequence_preview']}")

    elif ftype == "ligand":
        typer.echo(f"  Molecules: {info['num_molecules']}")
        for mol in info["molecules"][:5]:
            props = f", props={mol['properties']}" if mol["properties"] else ""
            typer.echo(
                f"    {mol['name']}: {mol['atoms']} atoms, "
                f"{mol['bonds']} bonds, elements={mol['elements']}{props}"
            )
        if info["num_molecules"] > 5:
            typer.echo(f"    ... and {info['num_molecules'] - 5} more")

    elif ftype == "alignment":
        typer.echo(f"  Depth: {info['depth']} sequences")
        typer.echo(f"  Query length: {info['query_length']} residues")
        if info.get("headers_preview"):
            typer.echo(f"  First headers: {info['headers_preview']}")
