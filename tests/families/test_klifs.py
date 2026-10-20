"""KLIFS master table loading and label normalisation."""

import pandas as pd

from molfun.families.kinase.klifs import load_klifs_table


def _write_master(path):
    pd.DataFrame(
        {
            "structure_ID": [10, 11, 12, 13],
            "pdb": ["1abc", "1abd", "1abe", "1abf"],
            "chain": ["A", "B", "A", "A"],
            "alt": ["A", None, None, "B"],
            "uniprot": ["P1", "P1", "P2", "P3"],
            "kinase": ["K1", "K1", "K2", "K3"],
            "family": ["F1", "F1", "F2", "F3"],
            "group": ["G1", "G1", "G2", "G3"],
            "species": ["Human", "Human", "Mouse", "Human"],
            "pocket": ["X" * 85] * 4,
            "resolution": [2.0, 2.5, 1.8, 3.0],
            "DFG": ["in", "out-like", "na", "out"],
            "aC_helix": ["in", "out", "in", "na"],
            "curated": [True, True, True, False],
        }
    ).to_csv(path, index=False)


def test_klifs_normalizes_labels(tmp_path):
    _write_master(tmp_path / "master.csv")

    table = load_klifs_table(tmp_path / "master.csv")

    assert list(table["structure_id"]) == ["10", "11", "12"]
    assert list(table["dfg"]) == ["in", "out", None]
    assert list(table["ac_helix"]) == ["in", "out", "in"]
    assert table.loc[table.structure_id == "11", "alt"].item() is None
    assert table.loc[table.structure_id == "10", "alt"].item() == "A"
