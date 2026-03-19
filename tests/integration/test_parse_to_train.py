"""
Integration test: Parse files → Dataset → Train.

Creates temporary PDB / SDF / A3M files, parses them with the parser
subsystem, and verifies the parse results can feed into a training loop.
"""

import pytest

from molfun.data.parsers import PDBParser, SDFParser, A3MParser, auto_parser
from molfun.training import HeadOnlyFinetune
from tests.integration.conftest import build_custom_model, make_loader


SAMPLE_PDB = """\
HEADER    TEST
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 10.00           C
ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00 10.00           C
ATOM      4  N   GLY A   2       4.000   5.000   6.000  1.00 10.00           N
ATOM      5  CA  GLY A   2       5.000   6.000   7.000  1.00 10.00           C
ATOM      6  C   GLY A   2       6.000   7.000   8.000  1.00 10.00           C
END
"""

SAMPLE_SDF = """\
Ethanol
     RDKit          3D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5400    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.3100    1.3300    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
M  END
$$$$
"""

SAMPLE_A3M = """\
>query
MKFLA
>hit1
MKaFLA
>hit2
MK-LA
"""


class TestParseToTrain:
    """Parse temporary files → extract features → feed to training."""

    def test_pdb_parse_and_extract(self, tmp_path):
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(SAMPLE_PDB)

        parser = PDBParser()
        result = parser.parse_file(str(pdb_file))

        assert result.sequence is not None
        assert len(result.sequence) >= 2
        assert result.all_atom_positions is not None
        assert result.all_atom_positions.shape[-1] == 3

    def test_sdf_parse(self, tmp_path):
        sdf_file = tmp_path / "test.sdf"
        sdf_file.write_text(SAMPLE_SDF)

        parser = SDFParser()
        molecules = parser.parse_file(str(sdf_file))

        assert len(molecules) >= 1
        mol = molecules[0]
        assert len(mol.atoms) == 3
        assert len(mol.bonds) == 2

    def test_a3m_parse(self, tmp_path):
        a3m_file = tmp_path / "test.a3m"
        a3m_file.write_text(SAMPLE_A3M)

        parser = A3MParser()
        result = parser.parse_file(str(a3m_file))

        assert result.sequences is not None
        assert len(result.sequences) >= 2

    def test_auto_parser_dispatch(self, tmp_path):
        pdb_file = tmp_path / "auto.pdb"
        pdb_file.write_text(SAMPLE_PDB)

        parser = auto_parser(str(pdb_file))
        assert isinstance(parser, PDBParser)

        result = parser.parse_file(str(pdb_file))
        assert result.sequence is not None

    def test_parsed_data_feeds_training(self, tmp_path):
        """Parse PDB → use seq length to build model → train 1 epoch."""
        pdb_file = tmp_path / "train.pdb"
        pdb_file.write_text(SAMPLE_PDB)

        parsed = PDBParser().parse_file(str(pdb_file))
        n_res = len(parsed.sequence)
        assert n_res >= 2

        model = build_custom_model()
        loader = make_loader(n=8, batch_size=4, seq_len=n_res)

        strategy = HeadOnlyFinetune(lr=1e-3, amp=False, loss_fn="mse")
        history = strategy.fit(model, loader, epochs=1, verbose=False)

        assert len(history) == 1
        assert history[0]["train_loss"] > 0
