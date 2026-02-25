"""
CLI commands for downloading PDB structures and MSAs.
"""

from __future__ import annotations
from pathlib import Path
from typing import Annotated, Optional

import typer


def fetch_pdb(
    ids: Annotated[
        Optional[list[str]],
        typer.Argument(help="PDB IDs to download (e.g. 1abc 2xyz)."),
    ] = None,
    pfam: Annotated[Optional[str], typer.Option(help="Pfam family (e.g. PF00069).")] = None,
    uniprot: Annotated[Optional[list[str]], typer.Option(help="UniProt accessions.")] = None,
    output: Annotated[Path, typer.Option("-o", help="Output directory.")] = Path("data/structures"),
    fmt: Annotated[str, typer.Option(help="Format: cif or pdb.")] = "cif",
    max_structures: Annotated[int, typer.Option(help="Max structures for family/uniprot queries.")] = 100,
    resolution: Annotated[float, typer.Option(help="Max resolution (Å) for family/uniprot.")] = 3.0,
):
    """Download PDB structures from RCSB by ID, Pfam family, or UniProt accession."""
    from molfun.data.sources.pdb import PDBFetcher

    fetcher = PDBFetcher(cache_dir=str(output), fmt=fmt)

    if pfam:
        typer.echo(f"Fetching structures for Pfam {pfam} (max {max_structures}, ≤{resolution}Å)...")
        paths = fetcher.fetch_by_family(pfam, max_structures=max_structures, resolution_max=resolution)
    elif uniprot:
        typer.echo(f"Fetching structures for {len(uniprot)} UniProt accessions...")
        paths = fetcher.fetch_by_uniprot(uniprot, max_per_accession=max_structures, resolution_max=resolution)
    elif ids:
        typer.echo(f"Fetching {len(ids)} PDB structure(s)...")
        paths = fetcher.fetch(ids)
    else:
        raise typer.BadParameter("Provide PDB IDs, --pfam, or --uniprot.")

    typer.echo(f"Downloaded {len(paths)} structure(s) to {output}/")
    for p in paths[:10]:
        typer.echo(f"  {p}")
    if len(paths) > 10:
        typer.echo(f"  ... and {len(paths) - 10} more")


def fetch_msa(
    sequence: Annotated[Optional[str], typer.Argument(help="Protein sequence or PDB ID.")] = None,
    ids_file: Annotated[Optional[Path], typer.Option("--ids", help="File with PDB IDs (one per line).")] = None,
    backend: Annotated[str, typer.Option(help="Backend: colabfold, precomputed, single.")] = "colabfold",
    msa_dir: Annotated[Path, typer.Option("-o", help="MSA output directory.")] = Path("data/msas"),
    max_depth: Annotated[int, typer.Option(help="Max MSA depth.")] = 512,
):
    """Fetch MSAs via ColabFold API or load precomputed A3M files."""
    from molfun.data.sources.msa import MSAProvider

    provider = MSAProvider(backend=backend, msa_dir=str(msa_dir), max_msa_depth=max_depth)

    if ids_file and ids_file.exists():
        entries = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
        typer.echo(f"Fetching MSAs for {len(entries)} entries via {backend}...")
        for pdb_id in entries:
            try:
                features = provider.get(pdb_id, pdb_id)
                depth = features["msa"].shape[0]
                length = features["msa"].shape[1]
                typer.echo(f"  {pdb_id}: depth={depth}, length={length}")
            except Exception as e:
                typer.echo(f"  {pdb_id}: FAILED — {e}", err=True)
    elif sequence:
        pdb_id = sequence if len(sequence) <= 6 else "query"
        typer.echo(f"Fetching MSA for {pdb_id} via {backend}...")
        features = provider.get(sequence, pdb_id)
        depth = features["msa"].shape[0]
        length = features["msa"].shape[1]
        typer.echo(f"  MSA: depth={depth}, length={length}")
    else:
        raise typer.BadParameter("Provide a sequence/PDB ID or --ids file.")

    typer.echo(f"MSAs saved to {msa_dir}/")
