"""
Curated protein collections for domain-specific fine-tuning.

Provides pre-built query recipes for common protein families and functional
groups. Each collection returns a dict of RCSB query parameters that can be
passed directly to ``PDBFetcher.fetch_combined`` or used as starting points
for custom queries.

Usage::

    from molfun.data.collections import COLLECTIONS, fetch_collection

    # See available collections
    print(list(COLLECTIONS))

    # Fetch all human kinases at ≤2.5 Å
    paths = fetch_collection("kinases_human", max_structures=200, resolution_max=2.5)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from molfun.data.sources.pdb import PDBFetcher


@dataclass
class CollectionSpec:
    """Defines a reusable protein collection query."""
    name: str
    description: str
    pfam_id: Optional[str] = None
    ec_number: Optional[str] = None
    go_id: Optional[str] = None
    taxonomy_id: Optional[int] = None
    keyword: Optional[str] = None
    uniprot_ids: Optional[list[str]] = None
    default_resolution: float = 3.0
    default_max: int = 500
    tags: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Curated collections
# ------------------------------------------------------------------

COLLECTIONS: dict[str, CollectionSpec] = {}


def _register(spec: CollectionSpec) -> None:
    COLLECTIONS[spec.name] = spec


# ── Kinases ──────────────────────────────────────────────────────────

_register(CollectionSpec(
    name="kinases",
    description="Protein kinases (Pfam PF00069 — Pkinase domain)",
    pfam_id="PF00069",
    default_resolution=3.0,
    tags=["enzyme", "kinase", "signaling"],
))

_register(CollectionSpec(
    name="kinases_human",
    description="Human protein kinases",
    pfam_id="PF00069",
    taxonomy_id=9606,
    default_resolution=2.5,
    tags=["enzyme", "kinase", "signaling", "human"],
))

_register(CollectionSpec(
    name="tyrosine_kinases",
    description="Tyrosine kinases (EC 2.7.10)",
    ec_number="2.7.10",
    default_resolution=3.0,
    tags=["enzyme", "kinase", "tyrosine_kinase"],
))

_register(CollectionSpec(
    name="serine_threonine_kinases",
    description="Serine/threonine kinases (EC 2.7.11)",
    ec_number="2.7.11",
    default_resolution=3.0,
    tags=["enzyme", "kinase", "serine_threonine_kinase"],
))

# ── Proteases ────────────────────────────────────────────────────────

_register(CollectionSpec(
    name="serine_proteases",
    description="Serine proteases (Pfam PF00089 — Trypsin)",
    pfam_id="PF00089",
    default_resolution=2.5,
    tags=["enzyme", "protease", "serine"],
))

_register(CollectionSpec(
    name="metalloproteases",
    description="Zinc metalloproteases (Pfam PF00557 — Metallopeptidase M24)",
    pfam_id="PF00557",
    default_resolution=3.0,
    tags=["enzyme", "protease", "metalloprotease"],
))

_register(CollectionSpec(
    name="cysteine_proteases",
    description="Cysteine proteases — Papain family (Pfam PF00112)",
    pfam_id="PF00112",
    default_resolution=3.0,
    tags=["enzyme", "protease", "cysteine"],
))

# ── GPCRs ────────────────────────────────────────────────────────────

_register(CollectionSpec(
    name="gpcr",
    description="G-protein coupled receptors (Pfam PF00001 — 7tm_1)",
    pfam_id="PF00001",
    default_resolution=3.5,
    default_max=300,
    tags=["receptor", "membrane", "gpcr"],
))

_register(CollectionSpec(
    name="gpcr_human",
    description="Human GPCRs",
    pfam_id="PF00001",
    taxonomy_id=9606,
    default_resolution=3.5,
    tags=["receptor", "membrane", "gpcr", "human"],
))

# ── Ion channels ─────────────────────────────────────────────────────

_register(CollectionSpec(
    name="ion_channels",
    description="Voltage-gated potassium channels (Pfam PF00520)",
    pfam_id="PF00520",
    default_resolution=3.5,
    tags=["channel", "membrane", "ion_channel"],
))

# ── Immunoglobulins ──────────────────────────────────────────────────

_register(CollectionSpec(
    name="immunoglobulins",
    description="Immunoglobulin domains (Pfam PF07654 — C1-set)",
    pfam_id="PF07654",
    default_resolution=3.0,
    tags=["immune", "antibody", "immunoglobulin"],
))

_register(CollectionSpec(
    name="nanobodies",
    description="Nanobody / VHH domains (Pfam PF07686 — V-set)",
    pfam_id="PF07686",
    keyword="nanobody",
    default_resolution=3.0,
    tags=["immune", "nanobody", "therapeutic"],
))

# ── Globins ──────────────────────────────────────────────────────────

_register(CollectionSpec(
    name="globins",
    description="Globin family (Pfam PF00042)",
    pfam_id="PF00042",
    default_resolution=2.5,
    tags=["oxygen_transport", "globin"],
))

# ── SH2 / SH3 domains ───────────────────────────────────────────────

_register(CollectionSpec(
    name="sh2_domains",
    description="SH2 phosphotyrosine-binding domains (Pfam PF00017)",
    pfam_id="PF00017",
    default_resolution=2.5,
    tags=["signaling", "sh2", "domain"],
))

_register(CollectionSpec(
    name="sh3_domains",
    description="SH3 domains (Pfam PF00018)",
    pfam_id="PF00018",
    default_resolution=2.5,
    tags=["signaling", "sh3", "domain"],
))

# ── Nucleic-acid binding ─────────────────────────────────────────────

_register(CollectionSpec(
    name="zinc_fingers",
    description="Zinc finger C2H2 domains (Pfam PF00096)",
    pfam_id="PF00096",
    default_resolution=3.0,
    tags=["transcription_factor", "zinc_finger", "dna_binding"],
))

_register(CollectionSpec(
    name="helicase",
    description="DEAD-box RNA helicases (Pfam PF00270)",
    pfam_id="PF00270",
    default_resolution=3.0,
    tags=["enzyme", "helicase", "rna"],
))

# ── Oxidoreductases ──────────────────────────────────────────────────

_register(CollectionSpec(
    name="p450",
    description="Cytochrome P450 enzymes (Pfam PF00067)",
    pfam_id="PF00067",
    default_resolution=3.0,
    tags=["enzyme", "oxidoreductase", "p450", "drug_metabolism"],
))

# ── Organism-specific ────────────────────────────────────────────────

_register(CollectionSpec(
    name="human_all",
    description="All human protein structures",
    taxonomy_id=9606,
    default_resolution=3.0,
    default_max=1000,
    tags=["human", "all"],
))

_register(CollectionSpec(
    name="sars_cov2",
    description="SARS-CoV-2 protein structures",
    taxonomy_id=2697049,
    default_resolution=3.5,
    default_max=500,
    tags=["virus", "covid", "sars"],
))

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def list_collections(tag: Optional[str] = None) -> list[CollectionSpec]:
    """
    List available collections, optionally filtered by tag.

    Args:
        tag: If given, return only collections containing this tag.
    """
    if tag is None:
        return list(COLLECTIONS.values())
    return [c for c in COLLECTIONS.values() if tag in c.tags]


def fetch_collection(
    name: str,
    *,
    cache_dir: Optional[str] = None,
    fmt: str = "cif",
    max_structures: Optional[int] = None,
    resolution_max: Optional[float] = None,
    deduplicate: bool = False,
    identity: float = 0.3,
    workers: int = 4,
    progress: bool = False,
    storage_options: Optional[dict] = None,
) -> list[str]:
    """
    Fetch structures for a named collection.

    Args:
        name: Collection name (see ``list_collections()``).
        cache_dir: Override cache directory.
        fmt: File format (``"cif"`` or ``"pdb"``).
        max_structures: Override collection default max.
        resolution_max: Override collection default resolution.
        deduplicate: If True, cluster by sequence and keep one representative.
        identity: Sequence identity threshold for deduplication.
        workers: Number of parallel download threads.
        progress: Show tqdm progress bar.
        storage_options: fsspec options for remote storage.

    Returns:
        List of file paths to downloaded structures.
    """
    if name not in COLLECTIONS:
        available = ", ".join(sorted(COLLECTIONS.keys()))
        raise ValueError(f"Unknown collection '{name}'. Available: {available}")

    spec = COLLECTIONS[name]
    fetcher = PDBFetcher(
        cache_dir=cache_dir, fmt=fmt, workers=workers,
        progress=progress, storage_options=storage_options,
    )

    max_s = max_structures or spec.default_max
    res = resolution_max or spec.default_resolution

    kwargs: dict = {}
    if spec.pfam_id:
        kwargs["pfam_id"] = spec.pfam_id
    if spec.ec_number:
        kwargs["ec_number"] = spec.ec_number
    if spec.go_id:
        kwargs["go_id"] = spec.go_id
    if spec.taxonomy_id:
        kwargs["taxonomy_id"] = spec.taxonomy_id
    if spec.keyword:
        kwargs["keyword"] = spec.keyword
    if spec.uniprot_ids:
        kwargs["uniprot_ids"] = spec.uniprot_ids

    if len(kwargs) > 1 or (len(kwargs) == 1 and spec.keyword):
        pdb_ids = fetcher.search_ids(
            **kwargs, max_results=max_s, resolution_max=res,
        )
    elif spec.pfam_id:
        pdb_ids = fetcher._search_rcsb(
            _build_simple_query(spec, res), max_results=max_s,
        )
    elif spec.ec_number:
        pdb_ids = fetcher._search_rcsb(
            _build_simple_query(spec, res), max_results=max_s,
        )
    elif spec.taxonomy_id:
        pdb_ids = fetcher._search_rcsb(
            _build_simple_query(spec, res), max_results=max_s,
        )
    elif spec.go_id:
        pdb_ids = fetcher._search_rcsb(
            _build_simple_query(spec, res), max_results=max_s,
        )
    else:
        raise ValueError(f"Collection '{name}' has no query criteria.")

    if deduplicate and pdb_ids:
        from molfun.data.sources.pdb import deduplicate_by_sequence
        pdb_ids = deduplicate_by_sequence(pdb_ids, identity=identity)

    return fetcher.fetch(pdb_ids)


def count_collection(
    name: str,
    resolution_max: Optional[float] = None,
) -> int:
    """
    Count how many structures are available in RCSB for a collection.

    Makes a single lightweight HTTP request (no downloads).

    Args:
        name: Collection name (see ``list_collections()``).
        resolution_max: Override collection default resolution.

    Returns:
        Total number of matching structures in RCSB.
    """
    if name not in COLLECTIONS:
        available = ", ".join(sorted(COLLECTIONS.keys()))
        raise ValueError(f"Unknown collection '{name}'. Available: {available}")

    spec = COLLECTIONS[name]
    res = resolution_max or spec.default_resolution
    fetcher = PDBFetcher()

    kwargs: dict = {}
    if spec.pfam_id:
        kwargs["pfam_id"] = spec.pfam_id
    if spec.ec_number:
        kwargs["ec_number"] = spec.ec_number
    if spec.go_id:
        kwargs["go_id"] = spec.go_id
    if spec.taxonomy_id:
        kwargs["taxonomy_id"] = spec.taxonomy_id
    if spec.keyword:
        kwargs["keyword"] = spec.keyword

    return fetcher.count(**kwargs, resolution_max=res)


def count_all_collections(
    resolution_max: Optional[float] = None,
    tag: Optional[str] = None,
) -> dict[str, int]:
    """
    Count available structures for all (or filtered) collections.

    Returns a dict of ``{collection_name: total_count}``.

    Args:
        resolution_max: Override default resolution for all queries.
        tag: If given, only count collections matching this tag.
    """
    specs = list_collections(tag=tag)
    counts: dict[str, int] = {}
    for spec in specs:
        try:
            counts[spec.name] = count_collection(
                spec.name, resolution_max=resolution_max,
            )
        except Exception:
            counts[spec.name] = -1
    return counts


def _build_simple_query(spec: CollectionSpec, resolution_max: float) -> dict:
    """Build an RCSB query for a collection with a single filter."""
    from molfun.data.sources.pdb import (
        _pfam_query, _ec_query, _go_query, _taxonomy_query, _keyword_query,
    )
    if spec.pfam_id:
        return _pfam_query(spec.pfam_id, resolution_max)
    if spec.ec_number:
        return _ec_query(spec.ec_number, resolution_max)
    if spec.go_id:
        return _go_query(spec.go_id, resolution_max)
    if spec.taxonomy_id:
        return _taxonomy_query(spec.taxonomy_id, resolution_max)
    if spec.keyword:
        return _keyword_query(spec.keyword, resolution_max)
    raise ValueError("No query criteria")
