"""
Abstract base parsers following Interface Segregation Principle.

Three parser families:
- BaseStructureParser: PDB/mmCIF → protein structure features
- BaseLigandParser: SDF/MOL2 → small molecule atoms + bonds
- BaseAlignmentParser: A3M/FASTA → sequence alignments

Each defines both ``parse_text()`` and ``parse_file()``.
Consumers depend on these abstractions, never on concrete parsers
(Dependency Inversion Principle).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch

from molfun.data.storage import open_path


# ======================================================================
# Parsed data structures (value objects)
# ======================================================================

@dataclass
class ParsedStructure:
    """Standardized output from any structure parser."""
    sequence: str
    residue_index: torch.Tensor       # [L]
    all_atom_positions: torch.Tensor   # [L, 3] (CA coords)
    all_atom_mask: torch.Tensor        # [L]
    seq_length: torch.Tensor           # [1]
    chain_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "sequence": self.sequence,
            "residue_index": self.residue_index,
            "all_atom_positions": self.all_atom_positions,
            "all_atom_mask": self.all_atom_mask,
            "seq_length": self.seq_length,
        }
        if self.chain_ids:
            d["chain_ids"] = self.chain_ids
        return d


@dataclass
class ParsedAtom:
    """Single atom in a molecule."""
    index: int
    element: str
    x: float
    y: float
    z: float
    charge: float = 0.0
    atom_type: str = ""
    name: str = ""
    residue: str = ""


@dataclass
class ParsedBond:
    """Bond between two atoms."""
    atom1: int
    atom2: int
    order: int = 1  # 1=single, 2=double, 3=triple, 4=aromatic


@dataclass
class ParsedMolecule:
    """Standardized output from any ligand parser."""
    name: str
    atoms: list[ParsedAtom]
    bonds: list[ParsedBond]
    properties: dict = field(default_factory=dict)

    @property
    def num_atoms(self) -> int:
        return len(self.atoms)

    @property
    def num_bonds(self) -> int:
        return len(self.bonds)

    @property
    def coords(self) -> torch.Tensor:
        """Atom coordinates as [N, 3] tensor."""
        return torch.tensor(
            [[a.x, a.y, a.z] for a in self.atoms],
            dtype=torch.float32,
        )

    @property
    def elements(self) -> list[str]:
        return [a.element for a in self.atoms]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "num_atoms": self.num_atoms,
            "num_bonds": self.num_bonds,
            "coords": self.coords,
            "elements": self.elements,
            "properties": self.properties,
        }


@dataclass
class ParsedAlignment:
    """Standardized output from any alignment parser."""
    msa: torch.Tensor            # [N, L] residue indices
    deletion_matrix: torch.Tensor  # [N, L]
    msa_mask: torch.Tensor        # [N, L]
    sequences: list[str] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return self.msa.shape[0]

    @property
    def length(self) -> int:
        return self.msa.shape[1]

    def to_dict(self) -> dict:
        return {
            "msa": self.msa,
            "deletion_matrix": self.deletion_matrix,
            "msa_mask": self.msa_mask,
        }


# ======================================================================
# Abstract base parsers (Interface Segregation)
# ======================================================================

class BaseStructureParser(ABC):
    """
    Parse macromolecular structure files.

    Subclasses: PDBParser, CIFParser.
    Single Responsibility: parse one format → ParsedStructure.
    """

    def __init__(self, max_seq_len: int = 512):
        self.max_seq_len = max_seq_len

    @abstractmethod
    def parse_text(self, text: str) -> ParsedStructure:
        """Parse from a string (file contents)."""
        ...

    def parse_file(
        self,
        path: str,
        storage_options: Optional[dict] = None,
    ) -> ParsedStructure:
        """Parse from a local or remote file path."""
        with open_path(path, "r", storage_options) as f:
            return self.parse_text(f.read())

    @staticmethod
    def extensions() -> list[str]:
        """File extensions this parser handles (e.g. ['.pdb'])."""
        return []


class BaseLigandParser(ABC):
    """
    Parse small molecule / ligand files.

    Subclasses: SDFParser, MOL2Parser.
    Single Responsibility: parse one format → ParsedMolecule(s).
    """

    @abstractmethod
    def parse_text(self, text: str) -> list[ParsedMolecule]:
        """Parse from string. Returns list (SDF can contain multiple mols)."""
        ...

    def parse_file(
        self,
        path: str,
        storage_options: Optional[dict] = None,
    ) -> list[ParsedMolecule]:
        """Parse from a local or remote file path."""
        with open_path(path, "r", storage_options) as f:
            return self.parse_text(f.read())

    @staticmethod
    def extensions() -> list[str]:
        return []


class BaseAlignmentParser(ABC):
    """
    Parse sequence alignment files.

    Subclasses: A3MParser, FASTAParser.
    Single Responsibility: parse one format → ParsedAlignment.
    """

    def __init__(self, max_depth: int = 512):
        self.max_depth = max_depth

    @abstractmethod
    def parse_text(self, text: str) -> ParsedAlignment:
        """Parse from string."""
        ...

    def parse_file(
        self,
        path: str,
        storage_options: Optional[dict] = None,
    ) -> ParsedAlignment:
        """Parse from a local or remote file path."""
        mode = "r"
        if str(path).endswith(".gz"):
            import gzip
            with open_path(path, "rb", storage_options) as f:
                text = gzip.decompress(f.read()).decode()
            return self.parse_text(text)
        with open_path(path, mode, storage_options) as f:
            return self.parse_text(f.read())

    @staticmethod
    def extensions() -> list[str]:
        return []
