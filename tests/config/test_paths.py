"""Tests for pilot path configuration read from the environment."""

from pathlib import Path

import pytest

from molfun.config.paths import pilot_paths


def test_pilot_paths_reads_env(tmp_path):
    env = {
        "MOLFUN_KINASE_DATA": str(tmp_path / "kin"),
        "MOLFUN_WORK_DIR": str(tmp_path / "work"),
        "MOLFUN_SIMPLEFOLD_CCD_RDB": str(tmp_path / "boltz" / "ccd.rdb"),
    }

    paths = pilot_paths(environ=env)

    assert paths.kinase_data == tmp_path / "kin"
    assert paths.work_dir == tmp_path / "work"
    assert paths.ccd_rdb == tmp_path / "boltz" / "ccd.rdb"
    assert isinstance(paths.kinase_data, Path)


def test_pilot_paths_missing_key_names_it(tmp_path):
    env = {
        "MOLFUN_WORK_DIR": str(tmp_path / "work"),
        "MOLFUN_SIMPLEFOLD_CCD_RDB": str(tmp_path / "ccd.rdb"),
    }

    with pytest.raises(RuntimeError, match="MOLFUN_KINASE_DATA"):
        pilot_paths(environ=env)


def test_pilot_paths_expands_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    env = {
        "MOLFUN_KINASE_DATA": "~/kin",
        "MOLFUN_WORK_DIR": "~/work",
        "MOLFUN_SIMPLEFOLD_CCD_RDB": "~/ccd.rdb",
    }

    paths = pilot_paths(environ=env)

    assert paths.kinase_data == tmp_path / "kin"
