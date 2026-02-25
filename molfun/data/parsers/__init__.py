"""
molfun.data.parsers — Unified parsers for biological file formats.

Each parser follows the Single Responsibility Principle: one format,
one class. All parsers implement a common base (Open/Closed + Liskov).
Consumers depend on abstractions (Dependency Inversion).

Structure parsers:
    PDBParser    — PDB format (no deps)
    CIFParser    — mmCIF format (requires BioPython)

Ligand parsers:
    SDFParser    — SDF/MOL V2000 (no deps)
    MOL2Parser   — Tripos MOL2 (no deps)

Alignment parsers:
    A3MParser    — A3M MSA format (no deps)
    FASTAParser  — FASTA sequences (no deps)

Usage::

    from molfun.data.parsers import auto_parser

    # Auto-detect format from extension
    parser = auto_parser("1abc.cif")
    result = parser.parse_file("1abc.cif")

    # Or use specific parsers
    from molfun.data.parsers import PDBParser, SDFParser, A3MParser

    structure = PDBParser().parse_file("1abc.pdb")
    molecules = SDFParser().parse_file("ligands.sdf")
    alignment = A3MParser().parse_file("msas/1abc.a3m")
"""

from molfun.data.parsers.base import (
    BaseStructureParser,
    BaseLigandParser,
    BaseAlignmentParser,
    ParsedStructure,
    ParsedMolecule,
    ParsedAtom,
    ParsedBond,
    ParsedAlignment,
)
from molfun.data.parsers.residue import (
    THREE_TO_ONE,
    ONE_TO_THREE,
    AA_TO_IDX,
    IDX_TO_AA,
    BACKBONE_ATOMS,
)
from molfun.data.parsers.pdb import PDBParser
from molfun.data.parsers.a3m import A3MParser
from molfun.data.parsers.fasta import FASTAParser
from molfun.data.parsers.sdf import SDFParser
from molfun.data.parsers.mol2 import MOL2Parser


__all__ = [
    # Base classes
    "BaseStructureParser",
    "BaseLigandParser",
    "BaseAlignmentParser",
    # Data classes
    "ParsedStructure",
    "ParsedMolecule",
    "ParsedAtom",
    "ParsedBond",
    "ParsedAlignment",
    # Constants
    "THREE_TO_ONE",
    "ONE_TO_THREE",
    "AA_TO_IDX",
    "IDX_TO_AA",
    "BACKBONE_ATOMS",
    # Parsers
    "PDBParser",
    "CIFParser",
    "A3MParser",
    "FASTAParser",
    "SDFParser",
    "MOL2Parser",
    # Factory
    "auto_parser",
    "PARSER_REGISTRY",
]


PARSER_REGISTRY: dict[str, type] = {}


def _build_registry():
    """Register all parsers by their file extensions."""
    for cls in (PDBParser, A3MParser, FASTAParser, SDFParser, MOL2Parser):
        for ext in cls.extensions():
            PARSER_REGISTRY[ext] = cls
    try:
        from molfun.data.parsers.mmcif import CIFParser
        for ext in CIFParser.extensions():
            PARSER_REGISTRY[ext] = CIFParser
    except ImportError:
        pass


_build_registry()


def auto_parser(path: str, **kwargs):
    """
    Return the appropriate parser for a file path.

    Uses the file extension to select the parser class.
    Raises ValueError if the format is not recognized.

    Args:
        path: File path (local or remote).
        **kwargs: Passed to the parser constructor.

    Returns:
        An instance of the matching parser.
    """
    path_lower = str(path).lower()

    for ext in sorted(PARSER_REGISTRY, key=len, reverse=True):
        if path_lower.endswith(ext):
            return PARSER_REGISTRY[ext](**kwargs)

    available = sorted(set(PARSER_REGISTRY.keys()))
    raise ValueError(
        f"No parser found for '{path}'. "
        f"Supported extensions: {available}"
    )


def _lazy_cifparser(name):
    if name == "CIFParser":
        from molfun.data.parsers.mmcif import CIFParser
        return CIFParser
    raise AttributeError(f"module has no attribute {name!r}")


def __getattr__(name):
    return _lazy_cifparser(name)
