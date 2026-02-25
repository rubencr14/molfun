"""
A3M alignment format parser.

A3M is FASTA-like: lowercase characters represent insertions
(counted as deletions in the deletion matrix).
"""

from __future__ import annotations

import torch

from molfun.data.parsers.base import BaseAlignmentParser, ParsedAlignment
from molfun.data.parsers.residue import AA_TO_IDX


class A3MParser(BaseAlignmentParser):
    """
    Parse A3M multiple sequence alignments → MSA tensors.

    Usage::

        parser = A3MParser(max_depth=512)
        alignment = parser.parse_file("msas/1abc.a3m")
        msa_dict = alignment.to_dict()  # {msa, deletion_matrix, msa_mask}
    """

    def parse_text(self, text: str) -> ParsedAlignment:
        sequences = []
        headers = []
        current_seq: list[str] = []
        current_header = ""

        for line in text.splitlines():
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    headers.append(current_header)
                current_seq = []
                current_header = line[1:].strip()
            else:
                current_seq.append(line.strip())

        if current_seq:
            sequences.append("".join(current_seq))
            headers.append(current_header)

        if not sequences:
            raise ValueError("Empty A3M: no sequences found.")

        sequences = sequences[:self.max_depth]
        headers = headers[:self.max_depth]
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
                row.append(AA_TO_IDX.get(c.upper(), 20))
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

        msa = torch.tensor(msa_rows, dtype=torch.long)
        deletion = torch.tensor(del_rows, dtype=torch.float32)
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
        return [".a3m", ".a3m.gz"]
