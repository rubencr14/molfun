"""
MSA (Multiple Sequence Alignment) provider.

Backends:
- "precomputed": loads .a3m files from disk
- "colabfold":   queries ColabFold MMseqs2 API (no local DB needed)
- "single":      dummy MSA with only the query sequence (for prototyping)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import time
import urllib.request
import urllib.error

import torch

from molfun.data.storage import open_path, exists, ensure_dir

_COLABFOLD_API = "https://api.colabfold.com"

_AA_TO_IDX = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7,
    "H": 8, "I": 9, "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15,
    "T": 16, "W": 17, "Y": 18, "V": 19, "-": 21, "X": 20,
}


class MSAProvider:
    """
    Generates or loads MSAs for protein sequences.

    Usage:
        # Pre-computed .a3m files
        msa = MSAProvider("precomputed", msa_dir="msas/")
        features = msa.get("MKFL...", "1abc")

        # ColabFold server (no local DB)
        msa = MSAProvider("colabfold")
        features = msa.get("MKFL...", "1abc")

        # Single-sequence dummy (fast prototyping)
        msa = MSAProvider("single")
        features = msa.get("MKFL...", "1abc")
    """

    def __init__(
        self,
        backend: str = "precomputed",
        msa_dir: Optional[str] = None,
        max_msa_depth: int = 512,
        storage_options: Optional[dict] = None,
    ):
        if backend not in ("precomputed", "colabfold", "single"):
            raise ValueError(f"Unknown backend: {backend}. Use 'precomputed', 'colabfold', or 'single'.")
        self.backend = backend
        self.msa_dir = str(msa_dir) if msa_dir else str(Path.home() / ".molfun" / "msa_cache")
        self.storage_options = storage_options
        ensure_dir(self.msa_dir, storage_options=storage_options)
        self.max_msa_depth = max_msa_depth

    def get(self, sequence: str, pdb_id: str) -> dict:
        """
        Return MSA features for a sequence.

        Returns dict with:
            "msa":              LongTensor  [N, L]  residue indices
            "deletion_matrix":  FloatTensor [N, L]  deletion counts
            "msa_mask":         FloatTensor [N, L]  1 for valid positions
        """
        cached = self._load_cached(pdb_id)
        if cached is not None:
            return cached

        if self.backend == "single":
            return self._single_sequence(sequence)

        if self.backend == "colabfold":
            a3m = self._query_colabfold(sequence)
        elif self.backend == "precomputed":
            a3m = self._load_a3m(pdb_id)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        features = self._parse_a3m(a3m, max_depth=self.max_msa_depth)
        self._save_cached(pdb_id, features)
        return features

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _single_sequence(self, sequence: str) -> dict:
        """Dummy MSA: just the query sequence. No alignment search."""
        L = len(sequence)
        msa = torch.tensor([[_AA_TO_IDX.get(aa, 20) for aa in sequence]], dtype=torch.long)
        return {
            "msa": msa,
            "deletion_matrix": torch.zeros(1, L),
            "msa_mask": torch.ones(1, L),
        }

    def _query_colabfold(self, sequence: str) -> str:
        """Query ColabFold MMseqs2 API → return raw A3M string."""
        submit_url = f"{_COLABFOLD_API}/ticket/msa"
        data = json.dumps({"q": f">query\n{sequence}", "mode": "env"}).encode()
        req = urllib.request.Request(
            submit_url, data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            ticket = json.loads(resp.read())

        ticket_id = ticket["id"]
        result_url = f"{_COLABFOLD_API}/result/msa/{ticket_id}"

        for _ in range(120):
            time.sleep(2)
            try:
                with urllib.request.urlopen(result_url, timeout=10) as resp:
                    if resp.status == 200:
                        result = json.loads(resp.read())
                        a3m = result.get("a3m", "")
                        if a3m:
                            return a3m
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue
                raise

        raise TimeoutError(f"ColabFold MSA search timed out for ticket {ticket_id}")

    def _load_a3m(self, pdb_id: str) -> str:
        """Load pre-computed A3M from local or remote storage."""
        for ext in (".a3m", ".a3m.gz"):
            path = f"{self.msa_dir}/{pdb_id}{ext}"
            if exists(path, self.storage_options):
                if ext.endswith(".gz"):
                    import gzip, io
                    with open_path(path, "rb", self.storage_options) as f:
                        return gzip.decompress(f.read()).decode()
                with open_path(path, "r", self.storage_options) as f:
                    return f.read()
        raise FileNotFoundError(
            f"No A3M file for '{pdb_id}' in {self.msa_dir}. "
            f"Expected {pdb_id}.a3m or {pdb_id}.a3m.gz"
        )

    # ------------------------------------------------------------------
    # A3M parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_a3m(a3m_string: str, max_depth: int = 512) -> dict:
        """
        Parse A3M format → MSA tensors.

        A3M: FASTA-like, lowercase = insertions (counted as deletions).
        """
        sequences = []
        current = []
        for line in a3m_string.splitlines():
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                current = []
            else:
                current.append(line.strip())
        if current:
            sequences.append("".join(current))

        if not sequences:
            raise ValueError("Empty A3M: no sequences found.")

        sequences = sequences[:max_depth]
        query_len = sum(1 for c in sequences[0] if c == c.upper() and c != "-")

        msa_rows = []
        del_rows = []

        for seq in sequences:
            row = []
            dels = []
            del_count = 0
            for c in seq:
                if c.islower():
                    del_count += 1
                    continue
                row.append(_AA_TO_IDX.get(c.upper(), 20))
                dels.append(del_count)
                del_count = 0

            if len(row) < query_len:
                row.extend([21] * (query_len - len(row)))
                dels.extend([0] * (query_len - len(dels)))
            elif len(row) > query_len:
                row = row[:query_len]
                dels = dels[:query_len]

            msa_rows.append(row)
            del_rows.append(dels)

        N = len(msa_rows)
        L = query_len

        msa = torch.tensor(msa_rows, dtype=torch.long)        # [N, L]
        deletion = torch.tensor(del_rows, dtype=torch.float32) # [N, L]
        mask = (msa != 21).float()                             # [N, L]

        return {
            "msa": msa,
            "deletion_matrix": deletion,
            "msa_mask": mask,
        }

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_path(self, pdb_id: str) -> str:
        return f"{self.msa_dir}/{pdb_id}.msa.pt"

    def _load_cached(self, pdb_id: str) -> Optional[dict]:
        path = self._cache_path(pdb_id)
        if exists(path, self.storage_options):
            with open_path(path, "rb", self.storage_options) as f:
                import io
                buf = io.BytesIO(f.read())
                return torch.load(buf, map_location="cpu", weights_only=True)
        return None

    def _save_cached(self, pdb_id: str, features: dict) -> None:
        import io
        buf = io.BytesIO()
        torch.save(features, buf)
        buf.seek(0)
        with open_path(self._cache_path(pdb_id), "wb", self.storage_options) as f:
            f.write(buf.read())
