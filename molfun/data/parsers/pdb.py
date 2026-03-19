"""
PDB format parser — no external dependencies.

Reads ATOM/HETATM lines from PDB format text and extracts
residue sequence, CA coordinates, and per-residue masks.
"""

from __future__ import annotations

import torch

from molfun.data.parsers.base import BaseStructureParser, ParsedStructure
from molfun.data.parsers.residue import THREE_TO_ONE


class PDBParser(BaseStructureParser):
    """
    Parse PDB format files without BioPython.

    Handles standard ATOM records (columns are fixed-width per PDB spec).
    For full mmCIF support, use CIFParser instead.

    Usage::

        parser = PDBParser(max_seq_len=512)
        structure = parser.parse_file("1abc.pdb")
        features = structure.to_dict()  # ready for dataset
    """

    def parse_text(self, text: str) -> ParsedStructure:
        residues: list[dict] = []
        seen: set[tuple[str, str]] = set()
        chains_seen: list[str] = []

        for line in text.splitlines():
            if not line.startswith("ATOM"):
                continue

            resname = line[17:20].strip()
            chain = line[21]
            resseq = line[22:26].strip()
            atom_name = line[12:16].strip()
            key = (chain, resseq)

            if key not in seen:
                seen.add(key)
                one_letter = THREE_TO_ONE.get(resname, "X")
                residues.append({"aa": one_letter, "ca": None, "chain": chain})
                if not chains_seen or chains_seen[-1] != chain:
                    chains_seen.append(chain)

            if atom_name == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                residues[-1]["ca"] = [x, y, z]

        L = min(len(residues), self.max_seq_len)
        residues = residues[:L]

        sequence = "".join(r["aa"] for r in residues)
        ca_coords = torch.zeros(L, 3)
        ca_mask = torch.zeros(L)

        for i, r in enumerate(residues):
            if r["ca"] is not None:
                ca_coords[i] = torch.tensor(r["ca"])
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
        return [".pdb", ".ent"]
