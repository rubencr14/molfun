"""Tests for the unified parser system."""

import pytest
import torch

from molfun.data.parsers import (
    auto_parser,
    PDBParser,
    A3MParser,
    FASTAParser,
    SDFParser,
    MOL2Parser,
    PARSER_REGISTRY,
)
from molfun.data.parsers.base import (
    ParsedStructure,
    ParsedMolecule,
    ParsedAlignment,
)
from molfun.data.parsers.residue import THREE_TO_ONE, AA_TO_IDX, IDX_TO_AA


# ======================================================================
# Test data fixtures
# ======================================================================

SAMPLE_PDB = """\
HEADER    TEST
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.500   2.500   3.500  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   3.000   4.000  1.00  0.00           C
ATOM      4  N   GLY A   2       3.000   4.000   5.000  1.00  0.00           N
ATOM      5  CA  GLY A   2       3.500   4.500   5.500  1.00  0.00           C
ATOM      6  N   VAL A   3       5.000   6.000   7.000  1.00  0.00           N
ATOM      7  CA  VAL A   3       5.500   6.500   7.500  1.00  0.00           C
END
"""

SAMPLE_A3M = """\
>query
MKFLA
>hit1
MKFLa
>hit2
MK-LA
"""

SAMPLE_FASTA = """\
>seq1
MKFLAGH
>seq2
MKFL
"""

SAMPLE_SDF = """\
aspirin
     RDKit          3D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5400    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.3100    1.3300    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  2  0
M  END
>  <MW>
180.157
$$$$
caffeine
     RDKit          3D

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.3400    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
M  END
$$$$
"""

SAMPLE_MOL2 = """\
@<TRIPOS>MOLECULE
test_mol
 3 2 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.3     1  LIG1        0.0000
      2 C2          1.5400    0.0000    0.0000 C.2     1  LIG1        0.1200
      3 O1          2.3100    1.3300    0.0000 O.2     1  LIG1       -0.2400
@<TRIPOS>BOND
     1     1     2    1
     2     2     3    2
"""


# ======================================================================
# Residue constants
# ======================================================================

class TestResidueConstants:
    def test_three_to_one_complete(self):
        assert len(THREE_TO_ONE) >= 20
        assert THREE_TO_ONE["ALA"] == "A"
        assert THREE_TO_ONE["TRP"] == "W"

    def test_aa_to_idx_complete(self):
        assert len(AA_TO_IDX) >= 22
        assert AA_TO_IDX["A"] == 0
        assert AA_TO_IDX["-"] == 21

    def test_idx_to_aa_inverse(self):
        for aa, idx in AA_TO_IDX.items():
            assert IDX_TO_AA[idx] == aa


# ======================================================================
# PDB Parser
# ======================================================================

class TestPDBParser:
    def test_parse_text(self):
        parser = PDBParser()
        result = parser.parse_text(SAMPLE_PDB)
        assert isinstance(result, ParsedStructure)
        assert result.sequence == "AGV"
        assert result.seq_length.item() == 3
        assert result.all_atom_positions.shape == (3, 3)
        assert result.all_atom_mask.sum().item() == 3.0

    def test_ca_coordinates(self):
        parser = PDBParser()
        result = parser.parse_text(SAMPLE_PDB)
        assert result.all_atom_positions[0, 0].item() == pytest.approx(1.5)
        assert result.all_atom_positions[0, 1].item() == pytest.approx(2.5)

    def test_max_seq_len(self):
        parser = PDBParser(max_seq_len=2)
        result = parser.parse_text(SAMPLE_PDB)
        assert result.seq_length.item() == 2
        assert result.sequence == "AG"

    def test_to_dict(self):
        parser = PDBParser()
        result = parser.parse_text(SAMPLE_PDB)
        d = result.to_dict()
        assert "sequence" in d
        assert "all_atom_positions" in d
        assert isinstance(d["all_atom_positions"], torch.Tensor)

    def test_chain_ids(self):
        parser = PDBParser()
        result = parser.parse_text(SAMPLE_PDB)
        assert result.chain_ids == ["A"]

    def test_extensions(self):
        assert ".pdb" in PDBParser.extensions()


# ======================================================================
# A3M Parser
# ======================================================================

class TestA3MParser:
    def test_parse_text(self):
        parser = A3MParser()
        result = parser.parse_text(SAMPLE_A3M)
        assert isinstance(result, ParsedAlignment)
        assert result.depth == 3
        assert result.length == 5

    def test_deletion_matrix(self):
        # 'a' in "MKFLa" is a trailing insertion with no subsequent column,
        # so it doesn't appear in deletion_matrix. Test with a mid-sequence case.
        a3m = ">q\nMKFLA\n>h\nMKaFLA\n"
        parser = A3MParser()
        result = parser.parse_text(a3m)
        assert result.deletion_matrix[1, 2].item() == 1.0  # 'a' before F

    def test_gap_handling(self):
        parser = A3MParser()
        result = parser.parse_text(SAMPLE_A3M)
        assert result.msa[2, 2].item() == 21  # gap '-' in hit2

    def test_max_depth(self):
        parser = A3MParser(max_depth=2)
        result = parser.parse_text(SAMPLE_A3M)
        assert result.depth == 2

    def test_headers(self):
        parser = A3MParser()
        result = parser.parse_text(SAMPLE_A3M)
        assert result.headers[0] == "query"
        assert result.headers[1] == "hit1"

    def test_to_dict(self):
        parser = A3MParser()
        d = parser.parse_text(SAMPLE_A3M).to_dict()
        assert "msa" in d
        assert "deletion_matrix" in d
        assert "msa_mask" in d


# ======================================================================
# FASTA Parser
# ======================================================================

class TestFASTAParser:
    def test_parse_text(self):
        parser = FASTAParser()
        result = parser.parse_text(SAMPLE_FASTA)
        assert isinstance(result, ParsedAlignment)
        assert result.depth == 2
        assert result.length == 7  # padded to max length

    def test_padding(self):
        parser = FASTAParser()
        result = parser.parse_text(SAMPLE_FASTA)
        assert result.msa[1, 4].item() == 21  # gap padding for shorter seq

    def test_headers(self):
        parser = FASTAParser()
        result = parser.parse_text(SAMPLE_FASTA)
        assert result.headers == ["seq1", "seq2"]


# ======================================================================
# SDF Parser
# ======================================================================

class TestSDFParser:
    def test_parse_multi_molecule(self):
        parser = SDFParser()
        mols = parser.parse_text(SAMPLE_SDF)
        assert len(mols) == 2

    def test_first_molecule(self):
        parser = SDFParser()
        mol = parser.parse_text(SAMPLE_SDF)[0]
        assert isinstance(mol, ParsedMolecule)
        assert mol.name == "aspirin"
        assert mol.num_atoms == 3
        assert mol.num_bonds == 2

    def test_coordinates(self):
        parser = SDFParser()
        mol = parser.parse_text(SAMPLE_SDF)[0]
        assert mol.coords.shape == (3, 3)
        assert mol.coords[0, 0].item() == pytest.approx(0.0)
        assert mol.coords[1, 0].item() == pytest.approx(1.54)

    def test_elements(self):
        parser = SDFParser()
        mol = parser.parse_text(SAMPLE_SDF)[0]
        assert mol.elements == ["C", "C", "O"]

    def test_bonds(self):
        parser = SDFParser()
        mol = parser.parse_text(SAMPLE_SDF)[0]
        assert mol.bonds[0].order == 1  # single
        assert mol.bonds[1].order == 2  # double
        assert mol.bonds[0].atom1 == 0
        assert mol.bonds[0].atom2 == 1

    def test_properties(self):
        parser = SDFParser()
        mol = parser.parse_text(SAMPLE_SDF)[0]
        assert "MW" in mol.properties
        assert mol.properties["MW"] == "180.157"

    def test_to_dict(self):
        parser = SDFParser()
        d = parser.parse_text(SAMPLE_SDF)[0].to_dict()
        assert "coords" in d
        assert "elements" in d

    def test_second_molecule(self):
        parser = SDFParser()
        mol = parser.parse_text(SAMPLE_SDF)[1]
        assert mol.name == "caffeine"
        assert mol.num_atoms == 2


# ======================================================================
# MOL2 Parser
# ======================================================================

class TestMOL2Parser:
    def test_parse(self):
        parser = MOL2Parser()
        mols = parser.parse_text(SAMPLE_MOL2)
        assert len(mols) == 1

    def test_atoms(self):
        parser = MOL2Parser()
        mol = parser.parse_text(SAMPLE_MOL2)[0]
        assert mol.name == "test_mol"
        assert mol.num_atoms == 3
        assert mol.elements == ["C", "C", "O"]

    def test_coordinates(self):
        parser = MOL2Parser()
        mol = parser.parse_text(SAMPLE_MOL2)[0]
        assert mol.coords.shape == (3, 3)
        assert mol.coords[2, 1].item() == pytest.approx(1.33)

    def test_bonds(self):
        parser = MOL2Parser()
        mol = parser.parse_text(SAMPLE_MOL2)[0]
        assert mol.num_bonds == 2
        assert mol.bonds[0].order == 1
        assert mol.bonds[1].order == 2

    def test_charges(self):
        parser = MOL2Parser()
        mol = parser.parse_text(SAMPLE_MOL2)[0]
        assert mol.atoms[0].charge == pytest.approx(0.0)
        assert mol.atoms[2].charge == pytest.approx(-0.24)

    def test_atom_types(self):
        parser = MOL2Parser()
        mol = parser.parse_text(SAMPLE_MOL2)[0]
        assert mol.atoms[0].atom_type == "C.3"
        assert mol.atoms[2].atom_type == "O.2"


# ======================================================================
# auto_parser factory + registry
# ======================================================================

class TestAutoParser:
    def test_pdb(self):
        p = auto_parser("protein.pdb")
        assert isinstance(p, PDBParser)

    def test_sdf(self):
        p = auto_parser("ligand.sdf")
        assert isinstance(p, SDFParser)

    def test_mol2(self):
        p = auto_parser("ligand.mol2")
        assert isinstance(p, MOL2Parser)

    def test_a3m(self):
        p = auto_parser("alignment.a3m")
        assert isinstance(p, A3MParser)

    def test_fasta(self):
        p = auto_parser("seqs.fasta")
        assert isinstance(p, FASTAParser)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="No parser found"):
            auto_parser("data.xyz")

    def test_kwargs_forwarded(self):
        p = auto_parser("protein.pdb", max_seq_len=64)
        assert p.max_seq_len == 64

    def test_registry_has_all_extensions(self):
        assert ".pdb" in PARSER_REGISTRY
        assert ".sdf" in PARSER_REGISTRY
        assert ".mol2" in PARSER_REGISTRY
        assert ".a3m" in PARSER_REGISTRY
        assert ".fasta" in PARSER_REGISTRY
