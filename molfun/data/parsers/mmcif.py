"""
mmCIF format parser — requires BioPython.

Uses BioPython's MMCIFParser for full mmCIF support including
multi-model structures, non-standard residues, and metadata.
"""

from __future__ import annotations
from typing import Optional
import tempfile
import os

import torch

from molfun.data.parsers.base import BaseStructureParser, ParsedStructure
from molfun.data.parsers.residue import THREE_TO_ONE


class CIFParser(BaseStructureParser):
    """
    Parse mmCIF files via BioPython.

    Falls back to PDBParser-style parsing for .pdb files if
    BioPython is available.

    Usage::

        parser = CIFParser(max_seq_len=512)
        structure = parser.parse_file("1abc.cif")
        features = structure.to_dict()
    """

    def __init__(self, max_seq_len: int = 512):
        super().__init__(max_seq_len=max_seq_len)
        try:
            from Bio.PDB import MMCIFParser as _BioMMCIF, PDBParser as _BioPDB
            self._mmcif_cls = _BioMMCIF
            self._pdb_cls = _BioPDB
        except ImportError:
            raise ImportError(
                "BioPython is required for CIFParser: pip install biopython"
            )

    def parse_text(self, text: str) -> ParsedStructure:
        """Parse mmCIF text (writes to temp file for BioPython)."""
        is_cif = text.lstrip().startswith("data_") or "loop_" in text[:500]
        suffix = ".cif" if is_cif else ".pdb"
        parser_cls = self._mmcif_cls if is_cif else self._pdb_cls

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False,
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        try:
            bp_parser = parser_cls(QUIET=True)
            structure = bp_parser.get_structure("s", tmp_path)
            return self._extract(structure)
        finally:
            os.unlink(tmp_path)

    def parse_file(
        self,
        path: str,
        storage_options: Optional[dict] = None,
    ) -> ParsedStructure:
        from molfun.data.storage import is_remote

        if is_remote(str(path)):
            return super().parse_file(path, storage_options)

        suffix = str(path).lower()
        if suffix.endswith(".cif"):
            bp_parser = self._mmcif_cls(QUIET=True)
        else:
            bp_parser = self._pdb_cls(QUIET=True)

        structure = bp_parser.get_structure("s", str(path))
        return self._extract(structure)

    def _extract(self, structure) -> ParsedStructure:
        model = structure[0]
        residues = []
        chains_seen = []

        for chain in model:
            chain_id = chain.id
            if not chains_seen or chains_seen[-1] != chain_id:
                chains_seen.append(chain_id)

            for res in chain:
                if res.id[0] != " ":
                    continue
                resname = res.get_resname().strip()
                one_letter = THREE_TO_ONE.get(resname, "X")
                ca = res["CA"].get_vector().get_array() if "CA" in res else None
                residues.append((one_letter, ca))

        L = min(len(residues), self.max_seq_len)
        residues = residues[:L]

        sequence = "".join(r[0] for r in residues)
        ca_coords = torch.zeros(L, 3)
        ca_mask = torch.zeros(L)

        for i, (_, ca) in enumerate(residues):
            if ca is not None:
                ca_coords[i] = torch.tensor(ca)
                ca_mask[i] = 1.0

        return ParsedStructure(
            sequence=sequence,
            residue_index=torch.arange(L),
            all_atom_positions=ca_coords,
            all_atom_mask=ca_mask,
            seq_length=torch.tensor([L]),
            chain_ids=chains_seen,
        )

    @staticmethod
    def extensions() -> list[str]:
        return [".cif", ".mmcif"]
