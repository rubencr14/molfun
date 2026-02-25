"""
FASTA format parser.

Parses plain FASTA sequences. For aligned sequences with
insertion/deletion info, use A3MParser instead.
"""

from __future__ import annotations

import torch

from molfun.data.parsers.base import BaseAlignmentParser, ParsedAlignment
from molfun.data.parsers.residue import AA_TO_IDX


class FASTAParser(BaseAlignmentParser):
    """
    Parse FASTA sequences → single-row ParsedAlignment.

    Multi-sequence FASTA is treated as a pre-aligned MSA (all
    sequences must have the same length after parsing, padded
    with gaps if needed).

    Usage::

        parser = FASTAParser()
        alignment = parser.parse_text(">query\\nMKFLAGHRT")
    """

    def parse_text(self, text: str) -> ParsedAlignment:
        sequences = []
        headers = []
        current_seq: list[str] = []
        current_header = ""

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    headers.append(current_header)
                current_seq = []
                current_header = line[1:].strip()
            else:
                current_seq.append(line)

        if current_seq:
            sequences.append("".join(current_seq))
            headers.append(current_header)

        if not sequences:
            raise ValueError("Empty FASTA: no sequences found.")

        sequences = sequences[:self.max_depth]
        headers = headers[:self.max_depth]

        max_len = max(len(s) for s in sequences)

        msa_rows = []
        for seq in sequences:
            row = [AA_TO_IDX.get(c.upper(), 20) for c in seq]
            if len(row) < max_len:
                row.extend([21] * (max_len - len(row)))
            msa_rows.append(row)

        msa = torch.tensor(msa_rows, dtype=torch.long)
        deletion = torch.zeros_like(msa, dtype=torch.float32)
        mask = (msa != 21).float()

        return ParsedAlignment(
            msa=msa,
            deletion_matrix=deletion,
            msa_mask=mask,
            sequences=sequences,
            headers=headers,
        )

    @staticmethod
    def extensions() -> list[str]:
        return [".fasta", ".fa", ".faa"]
