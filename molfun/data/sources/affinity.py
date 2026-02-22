"""
Affinity data fetcher.

Downloads binding affinity datasets (PDBbind) and returns
them as structured records ready for dataset construction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import csv
import re


_DEFAULT_CACHE = Path.home() / ".molfun" / "affinity_cache"


@dataclass
class AffinityRecord:
    """Single protein-ligand affinity entry."""
    pdb_id: str
    affinity: float             # -log(Kd/Ki) or kcal/mol depending on source
    affinity_unit: str = "pK"   # "pK" | "kcal"
    resolution: float = 0.0
    year: int = 0
    ligand_name: str = ""
    sequence: str = ""
    metadata: dict = field(default_factory=dict)


class AffinityFetcher:
    """
    Parse and serve binding affinity data.

    Supports:
    - PDBbind index files (v2016–v2020): provide the path to
      `INDEX_general_PL_data.{year}` or the refined set index.
    - CSV files with columns: pdb_id, affinity, [resolution, year, ...].

    Usage:
        fetcher = AffinityFetcher()

        # From PDBbind index file
        records = fetcher.from_pdbbind_index("path/to/INDEX_refined_data.2020")

        # From CSV
        records = fetcher.from_csv("my_dataset.csv")

        # Filter
        refined = fetcher.filter(records, resolution_max=2.5, min_year=2015)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @staticmethod
    def from_pdbbind_index(index_path: str) -> list[AffinityRecord]:
        """
        Parse a PDBbind INDEX file.

        Expected format (space-separated, lines starting with # are comments):
            PDB_code  resolution  release_year  -logKd/Ki  Kd/Ki/IC50=value  reference  ligand_name
        """
        records = []
        path = Path(index_path)
        if not path.exists():
            raise FileNotFoundError(f"PDBbind index not found: {path}")

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                record = _parse_pdbbind_line(line)
                if record is not None:
                    records.append(record)
        return records

    @staticmethod
    def from_csv(
        csv_path: str,
        pdb_col: str = "pdb_id",
        affinity_col: str = "affinity",
        resolution_col: str = "resolution",
        sequence_col: str = "sequence",
        delimiter: str = ",",
    ) -> list[AffinityRecord]:
        """
        Load affinity records from a CSV file.

        At minimum needs columns for pdb_id and affinity.
        """
        records = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                records.append(AffinityRecord(
                    pdb_id=row[pdb_col].strip().lower(),
                    affinity=float(row[affinity_col]),
                    resolution=float(row.get(resolution_col, 0) or 0),
                    sequence=row.get(sequence_col, "") or "",
                    metadata={k: v for k, v in row.items()
                              if k not in (pdb_col, affinity_col, resolution_col, sequence_col)},
                ))
        return records

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @staticmethod
    def filter(
        records: list[AffinityRecord],
        resolution_max: Optional[float] = None,
        min_year: Optional[int] = None,
        pdb_ids: Optional[set[str]] = None,
    ) -> list[AffinityRecord]:
        """Filter records by resolution, year, or PDB ID whitelist."""
        out = records
        if resolution_max is not None:
            out = [r for r in out if 0 < r.resolution <= resolution_max]
        if min_year is not None:
            out = [r for r in out if r.year >= min_year]
        if pdb_ids is not None:
            pdb_ids = {p.lower() for p in pdb_ids}
            out = [r for r in out if r.pdb_id in pdb_ids]
        return out

    @staticmethod
    def to_label_dict(records: list[AffinityRecord]) -> dict[str, float]:
        """Convert records to {pdb_id: affinity} dict for dataset construction."""
        return {r.pdb_id: r.affinity for r in records}


# ------------------------------------------------------------------
# Internal parsers
# ------------------------------------------------------------------

_PDBBIND_PATTERN = re.compile(
    r"^(\w{4})\s+"            # PDB code
    r"([\d.]+)\s+"            # resolution
    r"(\d{4})\s+"             # year
    r"([\d.]+)\s+"            # -logKd/Ki
    r"(\S+)\s+"               # Kd/Ki/IC50=value
    r"(.*)$"                  # rest (reference, ligand name)
)


def _parse_pdbbind_line(line: str) -> Optional[AffinityRecord]:
    m = _PDBBIND_PATTERN.match(line)
    if m is None:
        return None

    pdb_id = m.group(1).lower()
    resolution = float(m.group(2))
    year = int(m.group(3))
    affinity = float(m.group(4))
    rest = m.group(6).strip()

    ligand_name = ""
    parts = rest.rsplit("//", 1)
    if len(parts) == 2:
        ligand_name = parts[1].strip().strip("()")

    return AffinityRecord(
        pdb_id=pdb_id,
        affinity=affinity,
        affinity_unit="pK",
        resolution=resolution,
        year=year,
        ligand_name=ligand_name,
    )
