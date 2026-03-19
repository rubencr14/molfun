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
from molfun.data.parsers.a3m import A3MParser
from molfun.data.parsers.residue import AA_TO_IDX as _AA_TO_IDX_NEW

_COLABFOLD_API = "https://api.colabfold.com"

_AA_TO_IDX = _AA_TO_IDX_NEW


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
    # A3M parsing (delegated to A3MParser)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_a3m(a3m_string: str, max_depth: int = 512) -> dict:
        """Parse A3M format → MSA tensors via A3MParser."""
        parser = A3MParser(max_depth=max_depth)
        return parser.parse_text(a3m_string).to_dict()

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
