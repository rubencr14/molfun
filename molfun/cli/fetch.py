"""
CLI commands for downloading PDB structures, MSAs, and domain collections.
"""

from __future__ import annotations
from pathlib import Path
from typing import Annotated, Optional

import typer


def fetch_domain(
    collection: Annotated[
        Optional[str],
        typer.Argument(help="Collection name (e.g. kinases, gpcr, sars_cov2). Use --list to see all."),
    ] = None,
    *,
    list_collections: Annotated[bool, typer.Option("--list", help="List available collections.")] = False,
    tag: Annotated[Optional[str], typer.Option(help="Filter collections by tag.")] = None,
    pfam: Annotated[Optional[str], typer.Option(help="Pfam ID for custom query (e.g. PF00069).")] = None,
    ec: Annotated[Optional[str], typer.Option(help="EC number (e.g. 2.7.11.1 or 2.7.*).")] = None,
    go: Annotated[Optional[str], typer.Option(help="GO term (e.g. GO:0004672).")] = None,
    taxonomy: Annotated[Optional[int], typer.Option(help="NCBI taxonomy ID (e.g. 9606 for human).")] = None,
    keyword: Annotated[Optional[str], typer.Option(help="Free-text keyword.")] = None,
    output: Annotated[Path, typer.Option("-o", help="Output directory.")] = Path("data/structures"),
    fmt: Annotated[str, typer.Option(help="Format: cif or pdb.")] = "cif",
    max_structures: Annotated[int, typer.Option("--max", help="Max structures to download.")] = 500,
    resolution: Annotated[float, typer.Option(help="Max resolution (Angstrom).")] = 3.0,
    deduplicate: Annotated[bool, typer.Option("--dedup", help="Deduplicate by sequence identity.")] = False,
    identity: Annotated[float, typer.Option(help="Sequence identity threshold for dedup (0-1).")] = 0.3,
    metadata: Annotated[bool, typer.Option("--metadata", help="Save metadata JSON alongside structures.")] = False,
    workers: Annotated[int, typer.Option("--workers", "-w", help="Parallel download threads.")] = 4,
    progress: Annotated[bool, typer.Option("--progress", "-p", help="Show download progress bar.")] = True,
):
    """Fetch domain-specific protein structures from RCSB for fine-tuning."""
    if list_collections:
        from molfun.data.collections import list_collections as lc, count_collection
        specs = lc(tag=tag)
        if not specs:
            typer.echo("No collections found." + (f" (tag={tag})" if tag else ""))
            raise typer.Exit()
        typer.echo(f"\nAvailable collections ({len(specs)}):\n")
        typer.echo(f"  {'Name':30s} {'Structures':>12s}  Description")
        typer.echo(f"  {'─' * 30} {'─' * 12}  {'─' * 40}")
        for s in specs:
            try:
                n = count_collection(s.name)
                count_str = f"{n:,}"
            except Exception:
                count_str = "?"
            typer.echo(f"  {s.name:30s} {count_str:>12s}  {s.description}")
        typer.echo("")
        raise typer.Exit()

    from molfun.data.sources.pdb import PDBFetcher

    fetcher = PDBFetcher(cache_dir=str(output), fmt=fmt, workers=workers, progress=progress)

    if collection:
        from molfun.data.collections import fetch_collection, COLLECTIONS
        if collection not in COLLECTIONS:
            available = ", ".join(sorted(COLLECTIONS.keys()))
            typer.echo(f"Unknown collection '{collection}'. Available: {available}", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Fetching collection '{collection}' (max {max_structures}, ≤{resolution} Å, {workers} workers)...")
        paths = fetch_collection(
            collection,
            cache_dir=str(output),
            fmt=fmt,
            max_structures=max_structures,
            resolution_max=resolution,
            deduplicate=deduplicate,
            identity=identity,
            workers=workers,
            progress=progress,
        )
    else:
        has_filter = any([pfam, ec, go, taxonomy, keyword])
        if not has_filter:
            typer.echo("Provide a collection name or at least one filter (--pfam, --ec, --go, --taxonomy, --keyword).", err=True)
            typer.echo("Use --list to see available collections.")
            raise typer.Exit(code=1)

        typer.echo(f"Custom query (max {max_structures}, ≤{resolution} Å)...")
        filters = []
        if pfam:
            filters.append(f"pfam={pfam}")
        if ec:
            filters.append(f"ec={ec}")
        if go:
            filters.append(f"go={go}")
        if taxonomy:
            filters.append(f"taxonomy={taxonomy}")
        if keyword:
            filters.append(f"keyword='{keyword}'")
        typer.echo(f"  Filters: {', '.join(filters)}")

        pdb_ids = fetcher.search_ids(
            pfam_id=pfam,
            ec_number=ec,
            go_id=go,
            taxonomy_id=taxonomy,
            keyword=keyword,
            max_results=max_structures,
            resolution_max=resolution,
        )

        if deduplicate and pdb_ids:
            from molfun.data.sources.pdb import deduplicate_by_sequence
            original = len(pdb_ids)
            pdb_ids = deduplicate_by_sequence(pdb_ids, identity=identity)
            typer.echo(f"  Deduplication: {original} → {len(pdb_ids)} representatives (identity={identity})")

        paths = fetcher.fetch(pdb_ids)

    typer.echo(f"Downloaded {len(paths)} structure(s) to {output}/")
    for p in paths[:10]:
        typer.echo(f"  {p}")
    if len(paths) > 10:
        typer.echo(f"  ... and {len(paths) - 10} more")

    if metadata and paths:
        import json as _json
        typer.echo("Fetching metadata from RCSB...")
        pdb_ids_for_meta = [Path(p).stem for p in paths]
        records = fetcher.fetch_with_metadata(pdb_ids_for_meta)
        meta_path = output / "metadata.json"
        with open(meta_path, "w") as f:
            _json.dump(
                [
                    {
                        "pdb_id": r.pdb_id,
                        "resolution": r.resolution,
                        "method": r.method,
                        "organism": r.organism,
                        "organism_id": r.organism_id,
                        "title": r.title,
                        "sequence_length": r.sequence_length,
                        "ec_numbers": r.ec_numbers,
                        "pfam_ids": r.pfam_ids,
                        "deposition_date": r.deposition_date,
                        "has_ligand": r.has_ligand,
                    }
                    for r in records
                ],
                f,
                indent=2,
            )
        typer.echo(f"  Metadata saved to {meta_path}")


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
    workers: Annotated[int, typer.Option("--workers", "-w", help="Parallel download threads.")] = 4,
    progress: Annotated[bool, typer.Option("--progress", "-p", help="Show download progress bar.")] = True,
):
    """Download PDB structures from RCSB by ID, Pfam family, or UniProt accession."""
    from molfun.data.sources.pdb import PDBFetcher

    fetcher = PDBFetcher(cache_dir=str(output), fmt=fmt, workers=workers, progress=progress)

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
