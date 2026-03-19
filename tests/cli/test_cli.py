"""Tests for the Molfun CLI commands."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from molfun.cli import app

runner = CliRunner()

# Proper PDB fixed-width format (columns 31-54 for coords)
SAMPLE_PDB_LINE = (
    "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
    "ATOM      2  CA  GLY A   2       4.000   5.000   6.000  1.00  0.00           C\n"
    "END\n"
)


# ======================================================================
# Help / no-args
# ======================================================================

class TestCLIHelp:
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # Typer with no_args_is_help=True exits with code 0 or 2
        assert "structure" in result.output or "Usage" in result.output

    def test_structure_help(self):
        result = runner.invoke(app, ["structure", "--help"])
        assert result.exit_code == 0
        assert "Fine-tune" in result.output

    def test_affinity_help(self):
        result = runner.invoke(app, ["affinity", "--help"])
        assert result.exit_code == 0
        assert "affinity" in result.output.lower()

    def test_fetch_pdb_help(self):
        result = runner.invoke(app, ["fetch-pdb", "--help"])
        assert result.exit_code == 0
        assert "PDB" in result.output

    def test_fetch_msa_help(self):
        result = runner.invoke(app, ["fetch-msa", "--help"])
        assert result.exit_code == 0
        assert "MSA" in result.output

    def test_parse_help(self):
        result = runner.invoke(app, ["parse", "--help"])
        assert result.exit_code == 0
        assert "Parse" in result.output

    def test_registry_help(self):
        result = runner.invoke(app, ["registry", "--help"])
        assert result.exit_code == 0

    def test_agent_help(self):
        result = runner.invoke(app, ["agent", "--help"])
        assert result.exit_code == 0

    def test_eval_help(self):
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0

    def test_benchmark_help(self):
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0


# ======================================================================
# info
# ======================================================================

class TestInfo:
    def test_info_runs(self):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Python" in result.output
        assert "PyTorch" in result.output


# ======================================================================
# registry
# ======================================================================

class TestRegistry:
    def test_registry_all(self):
        result = runner.invoke(app, ["registry"])
        assert result.exit_code == 0
        assert "Attention" in result.output
        assert "Trunk blocks" in result.output

    def test_registry_attention(self):
        result = runner.invoke(app, ["registry", "attention"])
        assert result.exit_code == 0
        assert "standard" in result.output or "flash" in result.output

    def test_registry_parsers(self):
        result = runner.invoke(app, ["registry", "parsers"])
        assert result.exit_code == 0
        assert ".pdb" in result.output
        assert ".sdf" in result.output

    def test_registry_verbose(self):
        result = runner.invoke(app, ["registry", "attention", "--verbose"])
        assert result.exit_code == 0


# ======================================================================
# parse
# ======================================================================

class TestParse:
    def test_parse_pdb(self, tmp_path):
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(SAMPLE_PDB_LINE)
        result = runner.invoke(app, ["parse", str(pdb_file)])
        assert result.exit_code == 0
        assert "structure" in result.output
        assert "Residues" in result.output

    def test_parse_sdf(self, tmp_path):
        sdf_file = tmp_path / "test.sdf"
        sdf_file.write_text(
            "mol\n     test\n\n"
            "  2  1  0  0  0  0  0  0  0  0999 V2000\n"
            "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "    1.5400    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "  1  2  1  0\n"
            "M  END\n$$$$\n"
        )
        result = runner.invoke(app, ["parse", str(sdf_file)])
        assert result.exit_code == 0
        assert "ligand" in result.output
        assert "Molecules" in result.output

    def test_parse_a3m(self, tmp_path):
        a3m_file = tmp_path / "test.a3m"
        a3m_file.write_text(">query\nMKFLA\n>hit1\nMKFLA\n")
        result = runner.invoke(app, ["parse", str(a3m_file)])
        assert result.exit_code == 0
        assert "alignment" in result.output
        assert "Depth" in result.output

    def test_parse_fasta(self, tmp_path):
        fa_file = tmp_path / "test.fasta"
        fa_file.write_text(">seq1\nMKFLAGHRT\n>seq2\nMKFLA\n")
        result = runner.invoke(app, ["parse", str(fa_file)])
        assert result.exit_code == 0
        assert "alignment" in result.output

    def test_parse_mol2(self, tmp_path):
        mol2_file = tmp_path / "test.mol2"
        mol2_file.write_text(
            "@<TRIPOS>MOLECULE\ntest\n 2 1 0 0 0\nSMALL\n\n"
            "@<TRIPOS>ATOM\n"
            "      1 C1  0.0 0.0 0.0 C.3 1 LIG 0.0\n"
            "      2 O1  1.5 0.0 0.0 O.2 1 LIG 0.0\n"
            "@<TRIPOS>BOND\n"
            "     1     1     2    1\n"
        )
        result = runner.invoke(app, ["parse", str(mol2_file)])
        assert result.exit_code == 0
        assert "ligand" in result.output

    def test_parse_json_output(self, tmp_path):
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(SAMPLE_PDB_LINE)
        result = runner.invoke(app, ["parse", str(pdb_file), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert data[0]["type"] == "structure"

    def test_parse_multiple(self, tmp_path):
        f1 = tmp_path / "a.pdb"
        f1.write_text(SAMPLE_PDB_LINE)
        f2 = tmp_path / "b.pdb"
        f2.write_text(SAMPLE_PDB_LINE)
        result = runner.invoke(app, ["parse", str(f1), str(f2)])
        assert result.exit_code == 0
        assert result.output.count("structure") == 2

    def test_parse_missing_file(self):
        result = runner.invoke(app, ["parse", "/nonexistent/file.pdb"])
        assert result.exit_code == 0
        assert "not found" in result.output.lower()

    def test_parse_unknown_format(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("some data\n")
        result = runner.invoke(app, ["parse", str(f)])
        assert result.exit_code == 0
        assert "Unsupported" in result.output or "error" in result.output.lower()
