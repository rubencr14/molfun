"""Upstream SimpleFold preprocessing of chains extracted by molfun.

Needs redis-server and the Boltz ``ccd.rdb`` (``MOLFUN_SIMPLEFOLD_CCD_RDB``).
"""

import shutil
import socket
from pathlib import Path

import pytest

pytest.importorskip("simplefold")

from molfun.families.kinase.chains import extract_chain  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "fixtures" / "simplefold" / "raw"
CRAMBIN = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"


def _ccd_rdb():
    from molfun.config.paths import pilot_paths

    try:
        path = pilot_paths().ccd_rdb
    except RuntimeError:
        return None
    return path if path.is_file() else None


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


needs_redis = pytest.mark.skipif(
    shutil.which("redis-server") is None or _ccd_rdb() is None,
    reason="needs redis-server and MOLFUN_SIMPLEFOLD_CCD_RDB",
)


@needs_redis
def test_prepare_records_from_extracted_chain(tmp_path):
    from molfun.backends.simplefold.records import (
        SimpleFoldRecordDataset,
        prepare_simplefold_records,
    )

    (tmp_path / "cif").mkdir()
    info = extract_chain(RAW / "1crn.cif.gz", "A", tmp_path / "cif" / "7.cif")

    tokens = prepare_simplefold_records(
        tmp_path / "cif", tmp_path / "rec", _ccd_rdb(), port=_free_port(), num_processes=1
    )

    assert info.seqres == CRAMBIN
    assert (tokens / "tokens" / "7.pkl").is_file()
    dataset = SimpleFoldRecordDataset(tokens, tmp_path / "rec" / "targets", {"g": ["7"]})
    assert dataset[0]["aa_seq"] == CRAMBIN


def _broken_chain(src, out):
    """1CRN chain A with residue 10 moved 20 Å away: upstream's ConsecutiveCA
    quality filter marks such a chain invalid (no tokens after tokenization)."""
    import gemmi

    st = gemmi.read_structure(str(src))
    for atom in st[0][0][9]:
        atom.pos = gemmi.Position(atom.pos.x + 20.0, atom.pos.y, atom.pos.z)
    doc = st.make_mmcif_document()
    doc.write_file(str(out))


@needs_redis
def test_prepare_records_counts_chains_failing_upstream_filters(tmp_path):
    import json

    from molfun.backends.simplefold.records import prepare_simplefold_records

    (tmp_path / "cif").mkdir()
    extract_chain(RAW / "1crn.cif.gz", "A", tmp_path / "cif" / "7.cif")
    _broken_chain(tmp_path / "cif" / "7.cif", tmp_path / "cif" / "9.cif")

    prepare_simplefold_records(
        tmp_path / "cif", tmp_path / "rec", _ccd_rdb(), port=_free_port(), num_processes=1
    )

    report = json.loads((tmp_path / "rec" / "preprocess_report.json").read_text())
    assert report["n_tokenized"] == 2
    assert report["n_valid"] == 1
    assert report["invalid"] == ["9"]


@needs_redis
def test_dataset_skips_records_without_valid_chains(tmp_path):
    from molfun.backends.simplefold.records import (
        SimpleFoldRecordDataset,
        prepare_simplefold_records,
    )

    (tmp_path / "cif").mkdir()
    extract_chain(RAW / "1crn.cif.gz", "A", tmp_path / "cif" / "7.cif")
    _broken_chain(tmp_path / "cif" / "7.cif", tmp_path / "cif" / "9.cif")
    tokens = prepare_simplefold_records(
        tmp_path / "cif", tmp_path / "rec", _ccd_rdb(), port=_free_port(), num_processes=1
    )

    ds = SimpleFoldRecordDataset(
        tokens, tmp_path / "rec" / "targets", {"mixed": ["7", "9"], "only_invalid": ["9"]}
    )

    assert len(ds) == 1
    assert ds.group_members == [["7"]]
    assert ds.skipped_ids == ["9"]
    assert ds[0]["aa_seq"] == CRAMBIN


@needs_redis
def test_prepare_records_reports_unparsed_inputs(tmp_path):
    """Upstream prints a traceback per failed file and exits 0; molfun must
    count what was actually written instead of trusting the exit code."""
    import json

    from molfun.backends.simplefold.records import prepare_simplefold_records

    (tmp_path / "cif").mkdir()
    extract_chain(RAW / "1crn.cif.gz", "A", tmp_path / "cif" / "7.cif")
    (tmp_path / "cif" / "8.cif").write_text("data_broken\n_cell.length_a 1.0\n")

    prepare_simplefold_records(
        tmp_path / "cif", tmp_path / "rec", _ccd_rdb(), port=_free_port(), num_processes=1
    )

    report = json.loads((tmp_path / "rec" / "preprocess_report.json").read_text())
    assert report["n_inputs"] == 2
    assert report["n_tokenized"] == 1
    assert report["missing"] == ["8"]


@needs_redis
def test_prepare_records_raises_when_nothing_is_parsed(tmp_path):
    from molfun.backends.simplefold.records import prepare_simplefold_records

    (tmp_path / "cif").mkdir()
    (tmp_path / "cif" / "8.cif").write_text("data_broken\n_cell.length_a 1.0\n")

    with pytest.raises(RuntimeError, match="no records"):
        prepare_simplefold_records(
            tmp_path / "cif", tmp_path / "rec", _ccd_rdb(), port=_free_port(), num_processes=1
        )


def test_prepare_records_reports_missing_redis(tmp_path, monkeypatch):
    from molfun.backends.simplefold import records

    monkeypatch.setattr(records, "_redis_server_binary", lambda: None)

    with pytest.raises(FileNotFoundError, match="redis-server"):
        records.prepare_simplefold_records(tmp_path, tmp_path / "out", tmp_path / "ccd.rdb")
