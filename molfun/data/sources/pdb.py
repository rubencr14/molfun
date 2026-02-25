"""
PDB structure fetcher with local caching.

Downloads mmCIF files from RCSB PDB. Supports fetching by:
- PDB IDs
- Pfam family
- UniProt accessions
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import urllib.request
import urllib.error
import gzip
import shutil
import hashlib

from molfun.data.storage import is_remote, open_path, exists, ensure_dir, list_files


_DEFAULT_CACHE = Path.home() / ".molfun" / "pdb_cache"
_RCSB_DOWNLOAD = "https://files.rcsb.org/download"
_RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"


class PDBFetcher:
    """
    Download and cache PDB structures (mmCIF format) from RCSB.

    Usage:
        fetcher = PDBFetcher()

        # By IDs
        paths = fetcher.fetch(["1abc", "2xyz"])

        # By Pfam family
        paths = fetcher.fetch_by_family("PF00069", max_structures=100)

        # By UniProt
        paths = fetcher.fetch_by_uniprot(["P12345"])
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        fmt: str = "cif",
        storage_options: Optional[dict] = None,
    ):
        """
        Args:
            cache_dir: Directory to cache downloaded files.
                       Supports local paths or remote URIs (s3://, gs://).
                       Default: ~/.molfun/pdb_cache
            fmt: File format — "cif" (mmCIF, default) or "pdb".
            storage_options: fsspec options for remote cache_dir
                (e.g. ``{"endpoint_url": "http://localhost:9000"}`` for MinIO).
        """
        self.cache_dir = str(cache_dir) if cache_dir else str(_DEFAULT_CACHE)
        self.storage_options = storage_options
        ensure_dir(self.cache_dir, storage_options=storage_options)
        if fmt not in ("cif", "pdb"):
            raise ValueError(f"fmt must be 'cif' or 'pdb', got '{fmt}'")
        self.fmt = fmt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, pdb_ids: list[str]) -> list[str]:
        """
        Download structures by PDB ID.

        Returns list of file paths (same order as input).
        Paths may be local or remote depending on cache_dir.
        Skips download if file already cached.
        """
        paths = []
        for pdb_id in pdb_ids:
            pdb_id = pdb_id.strip().lower()
            path = self._cached_path(pdb_id)
            if not exists(path, self.storage_options):
                self._download(pdb_id, path)
            paths.append(path)
        return paths

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

    @staticmethod
    def _search_rcsb(query: dict, max_results: int = 500) -> list[str]:
        """Execute an RCSB Search API query, return list of PDB IDs."""
        query["request_options"] = {
            "return_type": "entry",
            "results_content_type": ["experimental"],
            "paginate": {"start": 0, "rows": max_results},
        }
        query["return_type"] = "entry"

        data = json.dumps(query).encode()
        req = urllib.request.Request(
            _RCSB_SEARCH,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError:
            return []

        return [
            hit["identifier"].lower()
            for hit in result.get("result_set", [])
        ]


# ------------------------------------------------------------------
# RCSB query builders
# ------------------------------------------------------------------

def _pfam_query(pfam_id: str, resolution_max: float) -> dict:
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_annotation.annotation_id",
                        "operator": "exact_match",
                        "value": pfam_id,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": resolution_max,
                    },
                },
            ],
        },
    }


def _uniprot_query(uniprot_id: str, resolution_max: float) -> dict:
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                        "operator": "exact_match",
                        "value": uniprot_id,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": resolution_max,
                    },
                },
            ],
        },
    }
