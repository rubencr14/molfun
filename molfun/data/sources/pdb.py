"""
PDB structure fetcher with local caching.

Downloads mmCIF files from RCSB PDB. Supports fetching by:
- PDB IDs
- Pfam family
- UniProt accessions
- EC number (enzyme classification)
- GO terms (gene ontology)
- SCOP/CATH classification
- Taxonomy (organism)
- Free-text keyword search

Also supports metadata retrieval via RCSB GraphQL API and
sequence-based deduplication via MMseqs2 clustering.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import logging
import time
import urllib.request
import urllib.error
import gzip
import shutil
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

from molfun.data.storage import is_remote, open_path, exists, ensure_dir, list_files


_DEFAULT_CACHE = Path.home() / ".molfun" / "pdb_cache"
_RCSB_DOWNLOAD = "https://files.rcsb.org/download"
_RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
_RCSB_GRAPHQL = "https://data.rcsb.org/graphql"


@dataclass
class StructureRecord:
    """Metadata for a single PDB structure, enriched from RCSB."""
    pdb_id: str
    path: str = ""
    resolution: Optional[float] = None
    method: str = ""
    organism: str = ""
    organism_id: Optional[int] = None
    title: str = ""
    sequence_length: Optional[int] = None
    ec_numbers: list[str] = field(default_factory=list)
    pfam_ids: list[str] = field(default_factory=list)
    deposition_date: str = ""
    has_ligand: bool = False


def _make_progress_bar(total: int, desc: str):
    """Create a tqdm progress bar if available, otherwise return None."""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="file")
    except ImportError:
        return None


class PDBFetcher:
    """
    Download and cache PDB structures (mmCIF format) from RCSB.

    Usage::

        fetcher = PDBFetcher()

        # By IDs
        paths = fetcher.fetch(["1abc", "2xyz"])

        # By Pfam family (e.g. protein kinases)
        paths = fetcher.fetch_by_family("PF00069", max_structures=100)

        # By EC number (e.g. all transferases → kinases)
        paths = fetcher.fetch_by_ec("2.7.*", max_structures=200)

        # By GO term (e.g. protein kinase activity)
        paths = fetcher.fetch_by_go("GO:0004672")

        # By organism (e.g. human only)
        paths = fetcher.fetch_by_taxonomy(9606, max_structures=100)

        # By keyword (free-text search)
        paths = fetcher.fetch_by_keyword("tyrosine kinase", max_structures=50)

        # With metadata + deduplication
        records = fetcher.fetch_with_metadata(["1abc", "2xyz"])
        paths = fetcher.fetch_deduplicated(pdb_ids, identity=0.3)
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        fmt: str = "cif",
        workers: int = 4,
        progress: bool = False,
        storage_options: Optional[dict] = None,
    ):
        """
        Args:
            cache_dir: Directory to cache downloaded files.
                       Supports local paths or remote URIs (s3://, gs://).
                       Default: ~/.molfun/pdb_cache
            fmt: File format — "cif" (mmCIF, default) or "pdb".
            workers: Default number of parallel download threads (1 = sequential).
            progress: Show tqdm progress bar by default (requires tqdm).
            storage_options: fsspec options for remote cache_dir
                (e.g. ``{"endpoint_url": "http://localhost:9000"}`` for MinIO).
        """
        self.cache_dir = str(cache_dir) if cache_dir else str(_DEFAULT_CACHE)
        self.storage_options = storage_options
        ensure_dir(self.cache_dir, storage_options=storage_options)
        if fmt not in ("cif", "pdb"):
            raise ValueError(f"fmt must be 'cif' or 'pdb', got '{fmt}'")
        self.fmt = fmt
        self.workers = workers
        self.progress = progress

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        pdb_ids: list[str],
        workers: Optional[int] = None,
        progress: Optional[bool] = None,
    ) -> list[str]:
        """
        Download structures by PDB ID.

        Args:
            pdb_ids: PDB IDs to download.
            workers: Number of parallel download threads (1 = sequential).
                     Defaults to the instance ``workers`` setting (4).
                     RCSB handles 4-8 concurrent connections well.
            progress: Show a ``tqdm`` progress bar (requires tqdm installed).
                      Defaults to the instance ``progress`` setting.

        Returns:
            List of file paths (same order as input).
            Paths may be local or remote depending on cache_dir.
            Skips download if file already cached.
        """
        w = workers if workers is not None else self.workers
        show = progress if progress is not None else self.progress

        id_path: dict[str, str] = {}
        to_download: list[tuple[str, str]] = []
        for pdb_id in pdb_ids:
            pid = pdb_id.strip().lower()
            path = self._cached_path(pid)
            id_path[pid] = path
            if not exists(path, self.storage_options):
                to_download.append((pid, path))

        if not to_download:
            return [id_path[pid.strip().lower()] for pid in pdb_ids]

        failed: list[tuple[str, Exception]] = []

        if w > 1 and len(to_download) > 1:
            iterator = self._parallel_download(to_download, w, show)
        else:
            iterator = self._sequential_download(to_download, show)

        for pid, exc in iterator:
            if exc is not None:
                failed.append((pid, exc))

        if failed:
            msg = "; ".join(f"{pid}: {e}" for pid, e in failed[:5])
            extra = f" (and {len(failed) - 5} more)" if len(failed) > 5 else ""
            logger.warning("Failed to download %d structure(s): %s%s", len(failed), msg, extra)

        return [id_path[pid.strip().lower()] for pid in pdb_ids]

    def _parallel_download(
        self,
        items: list[tuple[str, str]],
        workers: int,
        progress: bool,
    ):
        """Download items in parallel using a thread pool."""
        bar = _make_progress_bar(len(items), "Downloading") if progress else None
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._download_with_retry, pid, path): pid
                    for pid, path in items
                }
                for future in as_completed(futures):
                    pid = futures[future]
                    try:
                        future.result()
                        yield pid, None
                    except Exception as exc:
                        yield pid, exc
                    if bar is not None:
                        bar.update(1)
        finally:
            if bar is not None:
                bar.close()

    def _sequential_download(
        self,
        items: list[tuple[str, str]],
        progress: bool,
    ):
        """Download items one at a time."""
        bar = _make_progress_bar(len(items), "Downloading") if progress else None
        try:
            for pid, path in items:
                try:
                    self._download_with_retry(pid, path)
                    yield pid, None
                except Exception as exc:
                    yield pid, exc
                if bar is not None:
                    bar.update(1)
        finally:
            if bar is not None:
                bar.close()

    def fetch_by_family(
        self,
        pfam_id: str,
        max_structures: int = 500,
        resolution_max: float = 3.0,
    ) -> list[Path]:
        """
        Fetch PDB structures belonging to a Pfam family via RCSB Search API.

        Args:
            pfam_id: Pfam accession (e.g. "PF00069").
            max_structures: Maximum number of structures to retrieve.
            resolution_max: Filter by resolution (Angstrom).

        Returns:
            List of local file paths.
        """
        pdb_ids = self._search_rcsb(
            _pfam_query(pfam_id, resolution_max),
            max_results=max_structures,
        )
        return self.fetch(pdb_ids)

    def fetch_by_uniprot(
        self,
        uniprot_ids: list[str],
        max_per_accession: int = 50,
        resolution_max: float = 3.0,
    ) -> list[Path]:
        """
        Fetch PDB structures mapped to UniProt accessions via RCSB Search API.

        Args:
            uniprot_ids: List of UniProt accessions (e.g. ["P12345"]).
            max_per_accession: Max structures per UniProt ID.
            resolution_max: Filter by resolution (Angstrom).

        Returns:
            List of local file paths (deduplicated).
        """
        seen = set()
        all_ids = []
        for uniprot_id in uniprot_ids:
            pdb_ids = self._search_rcsb(
                _uniprot_query(uniprot_id, resolution_max),
                max_results=max_per_accession,
            )
            for pid in pdb_ids:
                if pid not in seen:
                    seen.add(pid)
                    all_ids.append(pid)
        return self.fetch(all_ids)

    def fetch_by_ec(
        self,
        ec_number: str,
        max_structures: int = 500,
        resolution_max: float = 3.0,
    ) -> list[str]:
        """
        Fetch structures by Enzyme Commission (EC) number.

        Supports wildcards: ``"2.7.*"`` matches all transferase kinases.

        Args:
            ec_number: Full or partial EC number (e.g. ``"2.7.11.1"`` or ``"2.7.*"``).
            max_structures: Maximum number of structures.
            resolution_max: Filter by resolution (Angstrom).
        """
        pdb_ids = self._search_rcsb(
            _ec_query(ec_number, resolution_max),
            max_results=max_structures,
        )
        return self.fetch(pdb_ids)

    def fetch_by_go(
        self,
        go_id: str,
        max_structures: int = 500,
        resolution_max: float = 3.0,
    ) -> list[str]:
        """
        Fetch structures annotated with a Gene Ontology (GO) term.

        Args:
            go_id: GO accession (e.g. ``"GO:0004672"`` for protein kinase activity).
            max_structures: Maximum number of structures.
            resolution_max: Filter by resolution (Angstrom).
        """
        pdb_ids = self._search_rcsb(
            _go_query(go_id, resolution_max),
            max_results=max_structures,
        )
        return self.fetch(pdb_ids)

    def fetch_by_taxonomy(
        self,
        taxonomy_id: int,
        max_structures: int = 500,
        resolution_max: float = 3.0,
    ) -> list[str]:
        """
        Fetch structures from a specific organism via NCBI taxonomy ID.

        Args:
            taxonomy_id: NCBI taxonomy ID (e.g. 9606 for *Homo sapiens*,
                         10090 for *Mus musculus*).
            max_structures: Maximum number of structures.
            resolution_max: Filter by resolution (Angstrom).
        """
        pdb_ids = self._search_rcsb(
            _taxonomy_query(taxonomy_id, resolution_max),
            max_results=max_structures,
        )
        return self.fetch(pdb_ids)

    def fetch_by_keyword(
        self,
        keyword: str,
        max_structures: int = 500,
        resolution_max: float = 3.0,
    ) -> list[str]:
        """
        Free-text search over RCSB metadata (title, abstract, etc.).

        Args:
            keyword: Search phrase (e.g. ``"tyrosine kinase"``).
            max_structures: Maximum number of structures.
            resolution_max: Filter by resolution (Angstrom).
        """
        pdb_ids = self._search_rcsb(
            _keyword_query(keyword, resolution_max),
            max_results=max_structures,
        )
        return self.fetch(pdb_ids)

    def fetch_by_scop(
        self,
        scop_id: str,
        max_structures: int = 500,
        resolution_max: float = 3.0,
    ) -> list[str]:
        """
        Fetch structures by SCOPe classification ID.

        Args:
            scop_id: SCOPe sunid or lineage string (e.g. ``"b.1.1.1"``).
            max_structures: Maximum number of structures.
            resolution_max: Filter by resolution (Angstrom).
        """
        pdb_ids = self._search_rcsb(
            _scop_query(scop_id, resolution_max),
            max_results=max_structures,
        )
        return self.fetch(pdb_ids)

    def fetch_combined(
        self,
        *,
        pfam_id: Optional[str] = None,
        ec_number: Optional[str] = None,
        go_id: Optional[str] = None,
        taxonomy_id: Optional[int] = None,
        keyword: Optional[str] = None,
        uniprot_ids: Optional[list[str]] = None,
        max_structures: int = 500,
        resolution_max: float = 3.0,
    ) -> list[str]:
        """
        Fetch structures matching ALL provided criteria (AND logic).

        Allows combining multiple filters in a single RCSB query for
        precise domain-specific datasets.

        Example::

            # Human protein kinases at ≤2.5 Å
            paths = fetcher.fetch_combined(
                pfam_id="PF00069",
                taxonomy_id=9606,
                resolution_max=2.5,
            )
        """
        nodes = []
        if pfam_id:
            nodes.append(_pfam_node(pfam_id))
        if ec_number:
            nodes.append(_ec_node(ec_number))
        if go_id:
            nodes.append(_go_node(go_id))
        if taxonomy_id:
            nodes.append(_taxonomy_node(taxonomy_id))
        if keyword:
            nodes.append(_keyword_node(keyword))
        if uniprot_ids:
            for uid in uniprot_ids:
                nodes.append(_uniprot_node(uid))
        nodes.append(_resolution_node(resolution_max))

        if len(nodes) < 2:
            raise ValueError("Provide at least one filter besides resolution.")

        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": nodes,
            }
        }
        pdb_ids = self._search_rcsb(query, max_results=max_structures)
        return self.fetch(pdb_ids)

    # ------------------------------------------------------------------
    # Metadata enrichment
    # ------------------------------------------------------------------

    def fetch_with_metadata(self, pdb_ids: list[str]) -> list[StructureRecord]:
        """
        Download structures and enrich them with RCSB metadata.

        Returns a list of ``StructureRecord`` with resolution, organism,
        EC numbers, Pfam IDs, etc. populated via the RCSB GraphQL API.
        """
        paths = self.fetch(pdb_ids)
        metadata = _fetch_metadata_graphql(pdb_ids)

        records = []
        for pid, path in zip(pdb_ids, paths):
            pid_upper = pid.strip().upper()
            meta = metadata.get(pid_upper, {})
            records.append(StructureRecord(
                pdb_id=pid.strip().lower(),
                path=path,
                resolution=meta.get("resolution"),
                method=meta.get("method", ""),
                organism=meta.get("organism", ""),
                organism_id=meta.get("organism_id"),
                title=meta.get("title", ""),
                sequence_length=meta.get("sequence_length"),
                ec_numbers=meta.get("ec_numbers", []),
                pfam_ids=meta.get("pfam_ids", []),
                deposition_date=meta.get("deposition_date", ""),
                has_ligand=meta.get("has_ligand", False),
            ))
        return records

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def search_ids(
        self,
        *,
        pfam_id: Optional[str] = None,
        ec_number: Optional[str] = None,
        go_id: Optional[str] = None,
        taxonomy_id: Optional[int] = None,
        keyword: Optional[str] = None,
        max_results: int = 500,
        resolution_max: float = 3.0,
    ) -> list[str]:
        """
        Like ``fetch_combined`` but returns PDB IDs without downloading.

        Useful for deduplication pipelines where you want to filter IDs
        before committing to downloads.
        """
        nodes = []
        if pfam_id:
            nodes.append(_pfam_node(pfam_id))
        if ec_number:
            nodes.append(_ec_node(ec_number))
        if go_id:
            nodes.append(_go_node(go_id))
        if taxonomy_id:
            nodes.append(_taxonomy_node(taxonomy_id))
        if keyword:
            nodes.append(_keyword_node(keyword))
        nodes.append(_resolution_node(resolution_max))

        if len(nodes) < 2:
            raise ValueError("Provide at least one filter besides resolution.")

        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": nodes,
            }
        }
        return self._search_rcsb(query, max_results=max_results)

    def fetch_deduplicated(
        self,
        pdb_ids: list[str],
        identity: float = 0.3,
        coverage: float = 0.8,
    ) -> list[str]:
        """
        Download structures and remove redundancy by sequence clustering.

        Uses MMseqs2 ``easy-cluster`` if available, otherwise falls back to
        a simple hash-based greedy approach using RCSB sequence data.

        Args:
            pdb_ids: PDB IDs to fetch.
            identity: Sequence identity threshold for clustering (0-1).
            coverage: Minimum coverage for clustering.

        Returns:
            Paths to representative structures (one per cluster).
        """
        representatives = deduplicate_by_sequence(
            pdb_ids, identity=identity, coverage=coverage,
        )
        return self.fetch(representatives)

    def list_cached(self) -> list[str]:
        """Return all cached structure files."""
        gz_files = list_files(f"{self.cache_dir}/*.{self.fmt}.gz", self.storage_options)
        plain_files = list_files(f"{self.cache_dir}/*.{self.fmt}", self.storage_options)
        return sorted(gz_files + plain_files)

    def clear_cache(self) -> int:
        """Remove all cached files. Returns number of files removed."""
        files = self.list_cached()
        for f in files:
            if is_remote(f):
                from molfun.data.storage import _get_fs
                fs, fs_path = _get_fs(f, self.storage_options)
                fs.rm(fs_path)
            else:
                Path(f).unlink()
        return len(files)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cached_path(self, pdb_id: str) -> str:
        sep = "/" if is_remote(self.cache_dir) else str(Path("/"))
        return f"{self.cache_dir}{sep}{pdb_id}.{self.fmt}"

    def _download(self, pdb_id: str, dest: str) -> None:
        ext = "cif" if self.fmt == "cif" else "pdb"
        url = f"{_RCSB_DOWNLOAD}/{pdb_id}.{ext}.gz"

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as tmp:
            tmp_gz = tmp.name
        try:
            urllib.request.urlretrieve(url, tmp_gz)
        except urllib.error.HTTPError as e:
            raise FileNotFoundError(f"PDB ID '{pdb_id}' not found on RCSB: {e}") from e

        with gzip.open(tmp_gz, "rb") as f_in:
            content = f_in.read()
        Path(tmp_gz).unlink()

        with open_path(dest, "wb", self.storage_options) as f_out:
            f_out.write(content)

    def _download_with_retry(
        self,
        pdb_id: str,
        dest: str,
        max_retries: int = 3,
    ) -> None:
        """Download with exponential backoff on transient HTTP errors."""
        for attempt in range(max_retries):
            try:
                self._download(pdb_id, dest)
                return
            except FileNotFoundError:
                raise
            except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
                retryable = isinstance(e, urllib.error.HTTPError) and e.code in (429, 500, 502, 503)
                if not retryable and isinstance(e, urllib.error.HTTPError):
                    raise
                if attempt < max_retries - 1:
                    wait = 2 ** attempt + 0.1
                    logger.debug("Retry %d/%d for %s (%.1fs): %s", attempt + 1, max_retries, pdb_id, wait, e)
                    time.sleep(wait)
                else:
                    raise

    @staticmethod
    def _search_rcsb(query: dict, max_results: int = 500) -> list[str]:
        """Execute an RCSB Search API query, return list of PDB IDs."""
        _, ids = PDBFetcher._search_rcsb_full(query, max_results)
        return ids

    @staticmethod
    def _search_rcsb_full(query: dict, max_results: int = 500) -> tuple[int, list[str]]:
        """Execute RCSB query, return (total_count, pdb_ids)."""
        import copy
        q = copy.deepcopy(query)
        q["request_options"] = {
            "return_type": "entry",
            "results_content_type": ["experimental"],
            "paginate": {"start": 0, "rows": max_results},
        }
        q["return_type"] = "entry"

        data = json.dumps(q).encode()
        req = urllib.request.Request(
            _RCSB_SEARCH,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError:
            return 0, []

        total = result.get("total_count", 0)
        ids = [hit["identifier"].lower() for hit in result.get("result_set", [])]
        return total, ids

    def count(
        self,
        *,
        pfam_id: Optional[str] = None,
        ec_number: Optional[str] = None,
        go_id: Optional[str] = None,
        taxonomy_id: Optional[int] = None,
        keyword: Optional[str] = None,
        resolution_max: float = 3.0,
    ) -> int:
        """
        Count how many structures match a query **without downloading**.

        Uses the RCSB ``total_count`` field — a single cheap HTTP request.

        Example::

            n = fetcher.count(pfam_id="PF00069")
            print(f"Kinases available: {n}")
        """
        nodes = []
        if pfam_id:
            nodes.append(_pfam_node(pfam_id))
        if ec_number:
            nodes.append(_ec_node(ec_number))
        if go_id:
            nodes.append(_go_node(go_id))
        if taxonomy_id:
            nodes.append(_taxonomy_node(taxonomy_id))
        if keyword:
            nodes.append(_keyword_node(keyword))
        nodes.append(_resolution_node(resolution_max))

        if len(nodes) < 2:
            raise ValueError("Provide at least one filter besides resolution.")

        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": nodes,
            }
        }
        total, _ = self._search_rcsb_full(query, max_results=0)
        return total


# ------------------------------------------------------------------
# RCSB query node builders (composable)
# ------------------------------------------------------------------

def _resolution_node(resolution_max: float) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined",
            "operator": "less_or_equal",
            "value": resolution_max,
        },
    }


def _pfam_node(pfam_id: str) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "rcsb_polymer_entity_annotation.annotation_id",
            "operator": "exact_match",
            "value": pfam_id,
        },
    }


def _uniprot_node(uniprot_id: str) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": (
                "rcsb_polymer_entity_container_identifiers"
                ".reference_sequence_identifiers.database_accession"
            ),
            "operator": "exact_match",
            "value": uniprot_id,
        },
    }


def _ec_node(ec_number: str) -> dict:
    ec_clean = ec_number.rstrip(".*")
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "rcsb_polymer_entity.rcsb_ec_lineage.id",
            "operator": "exact_match",
            "value": ec_clean,
        },
    }


def _go_node(go_id: str) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": (
                "rcsb_polymer_entity_annotation"
                ".annotation_lineage.id"
            ),
            "operator": "exact_match",
            "value": go_id,
        },
    }


def _taxonomy_node(taxonomy_id: int) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "rcsb_entity_source_organism.taxonomy_lineage.id",
            "operator": "exact_match",
            "value": str(taxonomy_id),
        },
    }


def _keyword_node(keyword: str) -> dict:
    return {
        "type": "terminal",
        "service": "full_text",
        "parameters": {
            "value": keyword,
        },
    }


def _scop_node(scop_id: str) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": (
                "rcsb_polymer_entity_annotation"
                ".annotation_lineage.id"
            ),
            "operator": "exact_match",
            "value": scop_id,
        },
    }


# ------------------------------------------------------------------
# Full query builders (wrap nodes + resolution)
# ------------------------------------------------------------------

def _and_query(*nodes: dict) -> dict:
    return {"query": {"type": "group", "logical_operator": "and", "nodes": list(nodes)}}


def _pfam_query(pfam_id: str, resolution_max: float) -> dict:
    return _and_query(_pfam_node(pfam_id), _resolution_node(resolution_max))


def _uniprot_query(uniprot_id: str, resolution_max: float) -> dict:
    return _and_query(_uniprot_node(uniprot_id), _resolution_node(resolution_max))


def _ec_query(ec_number: str, resolution_max: float) -> dict:
    return _and_query(_ec_node(ec_number), _resolution_node(resolution_max))


def _go_query(go_id: str, resolution_max: float) -> dict:
    return _and_query(_go_node(go_id), _resolution_node(resolution_max))


def _taxonomy_query(taxonomy_id: int, resolution_max: float) -> dict:
    return _and_query(_taxonomy_node(taxonomy_id), _resolution_node(resolution_max))


def _keyword_query(keyword: str, resolution_max: float) -> dict:
    return _and_query(_keyword_node(keyword), _resolution_node(resolution_max))


def _scop_query(scop_id: str, resolution_max: float) -> dict:
    return _and_query(_scop_node(scop_id), _resolution_node(resolution_max))


# ------------------------------------------------------------------
# RCSB GraphQL metadata fetcher
# ------------------------------------------------------------------

_GRAPHQL_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    rcsb_entry_info {
      resolution_combined
      experimental_method
    }
    rcsb_entry_container_identifiers { entry_id }
    polymer_entities {
      rcsb_polymer_entity {
        rcsb_ec_lineage { id }
      }
      entity_poly { pdbx_seq_one_letter_code_can }
      rcsb_entity_source_organism {
        ncbi_scientific_name
        ncbi_taxonomy_id
      }
      rcsb_polymer_entity_annotation {
        annotation_id
        type
      }
    }
    nonpolymer_entities { rcsb_nonpolymer_entity { formula_weight } }
    rcsb_accession_info { deposit_date }
  }
}
"""


def _fetch_metadata_graphql(pdb_ids: list[str]) -> dict[str, dict]:
    """
    Fetch rich metadata for a batch of PDB IDs via RCSB GraphQL.

    Returns dict keyed by uppercase PDB ID with fields:
    resolution, method, organism, organism_id, title, sequence_length,
    ec_numbers, pfam_ids, deposition_date, has_ligand.
    """
    ids_upper = [pid.strip().upper() for pid in pdb_ids]
    result: dict[str, dict] = {}

    batch_size = 50
    for i in range(0, len(ids_upper), batch_size):
        batch = ids_upper[i : i + batch_size]
        payload = json.dumps({"query": _GRAPHQL_QUERY, "variables": {"ids": batch}}).encode()
        req = urllib.request.Request(
            _RCSB_GRAPHQL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue

        for entry in data.get("data", {}).get("entries", []) or []:
            if entry is None:
                continue
            pdb_id = entry.get("rcsb_id", "")
            info = entry.get("rcsb_entry_info") or {}
            poly = entry.get("polymer_entities") or []
            nonpoly = entry.get("nonpolymer_entities") or []
            accession = entry.get("rcsb_accession_info") or {}

            ec_nums = []
            pfam_ids = []
            organism = ""
            organism_id = None
            seq_len = 0
            for pe in poly:
                rpe = pe.get("rcsb_polymer_entity") or {}
                for ec in rpe.get("rcsb_ec_lineage") or []:
                    ec_id = ec.get("id", "")
                    if ec_id and ec_id not in ec_nums:
                        ec_nums.append(ec_id)
                for ann in pe.get("rcsb_polymer_entity_annotation") or []:
                    if ann.get("type") == "Pfam":
                        aid = ann.get("annotation_id", "")
                        if aid and aid not in pfam_ids:
                            pfam_ids.append(aid)
                seq = (pe.get("entity_poly") or {}).get(
                    "pdbx_seq_one_letter_code_can", ""
                )
                seq_len = max(seq_len, len(seq))
                for src in pe.get("rcsb_entity_source_organism") or []:
                    if not organism:
                        organism = src.get("ncbi_scientific_name", "")
                        organism_id = src.get("ncbi_taxonomy_id")

            res_combined = info.get("resolution_combined")
            resolution = res_combined[0] if isinstance(res_combined, list) and res_combined else res_combined

            result[pdb_id] = {
                "resolution": resolution,
                "method": info.get("experimental_method", ""),
                "organism": organism,
                "organism_id": organism_id,
                "title": (entry.get("struct") or {}).get("title", ""),
                "sequence_length": seq_len or None,
                "ec_numbers": ec_nums,
                "pfam_ids": pfam_ids,
                "deposition_date": accession.get("deposit_date", ""),
                "has_ligand": len(nonpoly) > 0,
            }

    return result


# ------------------------------------------------------------------
# Sequence deduplication
# ------------------------------------------------------------------

def _fetch_sequences_rcsb(pdb_ids: list[str]) -> dict[str, str]:
    """Fetch canonical sequences from RCSB GraphQL (first entity only)."""
    ids_upper = [pid.strip().upper() for pid in pdb_ids]
    seqs: dict[str, str] = {}

    gql = """
    query($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        polymer_entities {
          entity_poly { pdbx_seq_one_letter_code_can }
        }
      }
    }
    """
    batch_size = 50
    for i in range(0, len(ids_upper), batch_size):
        batch = ids_upper[i : i + batch_size]
        payload = json.dumps({"query": gql, "variables": {"ids": batch}}).encode()
        req = urllib.request.Request(
            _RCSB_GRAPHQL, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue
        for entry in data.get("data", {}).get("entries", []) or []:
            if entry is None:
                continue
            pid = entry["rcsb_id"].lower()
            for pe in entry.get("polymer_entities") or []:
                seq = (pe.get("entity_poly") or {}).get(
                    "pdbx_seq_one_letter_code_can", ""
                )
                if seq:
                    seqs[pid] = seq
                    break
    return seqs


def _mmseqs_available() -> bool:
    import subprocess
    try:
        subprocess.run(["mmseqs", "version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _cluster_mmseqs(fasta_path: str, identity: float, coverage: float) -> list[str]:
    """Run MMseqs2 easy-cluster, return representative IDs."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "clust"
        subprocess.run(
            [
                "mmseqs", "easy-cluster",
                fasta_path, str(prefix), tmpdir,
                "--min-seq-id", str(identity),
                "-c", str(coverage),
                "--cov-mode", "0",
            ],
            capture_output=True,
            check=True,
        )
        rep_file = Path(f"{prefix}_rep_seq.fasta")
        reps = []
        for line in rep_file.read_text().splitlines():
            if line.startswith(">"):
                reps.append(line[1:].strip().split()[0])
        return reps


def _cluster_greedy(sequences: dict[str, str], identity: float) -> list[str]:
    """
    Greedy sequence identity clustering (fallback when MMseqs2 is absent).

    Uses a simple k-mer Jaccard similarity as a fast proxy for identity.
    """
    k = 3
    kmer_sets: dict[str, set[str]] = {}
    for pid, seq in sequences.items():
        kmers = {seq[i : i + k] for i in range(len(seq) - k + 1)} if len(seq) >= k else {seq}
        kmer_sets[pid] = kmers

    sorted_ids = sorted(sequences, key=lambda p: len(sequences[p]), reverse=True)
    representatives: list[str] = []
    assigned: set[str] = set()

    for pid in sorted_ids:
        if pid in assigned:
            continue
        representatives.append(pid)
        assigned.add(pid)
        for other in sorted_ids:
            if other in assigned:
                continue
            inter = len(kmer_sets[pid] & kmer_sets[other])
            union = len(kmer_sets[pid] | kmer_sets[other])
            if union > 0 and inter / union >= identity:
                assigned.add(other)

    return representatives


def deduplicate_by_sequence(
    pdb_ids: list[str],
    identity: float = 0.3,
    coverage: float = 0.8,
) -> list[str]:
    """
    Cluster PDB IDs by sequence identity and return one representative per cluster.

    Uses MMseqs2 if installed, otherwise falls back to greedy k-mer clustering.

    Args:
        pdb_ids: PDB IDs to deduplicate.
        identity: Sequence identity threshold (0-1).
        coverage: Minimum alignment coverage (MMseqs2 only).

    Returns:
        List of representative PDB IDs.
    """
    if len(pdb_ids) <= 1:
        return list(pdb_ids)

    sequences = _fetch_sequences_rcsb(pdb_ids)
    if not sequences:
        return list(pdb_ids)

    if _mmseqs_available():
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            for pid, seq in sequences.items():
                f.write(f">{pid}\n{seq}\n")
            fasta_path = f.name
        try:
            reps = _cluster_mmseqs(fasta_path, identity, coverage)
        finally:
            Path(fasta_path).unlink(missing_ok=True)
    else:
        reps = _cluster_greedy(sequences, identity)

    return reps
