"""Tests for the unified storage abstraction (local filesystem)."""

import tempfile
from pathlib import Path
import pytest

from molfun.data.storage import open_path, list_files, exists, ensure_dir, is_remote


class TestIsRemote:
    def test_local_paths(self):
        assert not is_remote("/tmp/test.csv")
        assert not is_remote("data/file.txt")
        assert not is_remote("~/test.csv")
        assert not is_remote("file.csv")

    def test_remote_paths(self):
        assert is_remote("s3://bucket/file.csv")
        assert is_remote("gs://bucket/file.csv")
        assert is_remote("az://container/file.csv")
        assert is_remote("http://example.com/file.csv")


class TestOpenPathLocal:
    def test_read_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/test.txt"
            with open_path(path, "w") as f:
                f.write("hello molfun")
            with open_path(path, "r") as f:
                assert f.read() == "hello molfun"

    def test_binary_read_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/test.bin"
            data = b"\x00\x01\x02\x03"
            with open_path(path, "wb") as f:
                f.write(data)
            with open_path(path, "rb") as f:
                assert f.read() == data

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/nested/deep/file.txt"
            with open_path(path, "w") as f:
                f.write("nested")
            assert Path(path).exists()

    def test_tilde_expansion(self):
        path = "~/nonexistent_molfun_test_file_xyz.txt"
        expanded = Path(path).expanduser()
        assert not expanded.exists()
        assert not is_remote(path)


class TestListFilesLocal:
    def test_glob(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["a.csv", "b.csv", "c.txt"]:
                Path(tmpdir, name).write_text("data")
            csv_files = list_files(f"{tmpdir}/*.csv")
            assert len(csv_files) == 2
            assert all(f.endswith(".csv") for f in csv_files)

    def test_no_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert list_files(f"{tmpdir}/*.xyz") == []


class TestExistsLocal:
    def test_exists(self):
        with tempfile.NamedTemporaryFile() as tmp:
            assert exists(tmp.name)

    def test_not_exists(self):
        assert not exists("/tmp/nonexistent_molfun_xyz_123")


class TestEnsureDirLocal:
    def test_creates_nested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/a/b/c"
            ensure_dir(path)
            assert Path(path).is_dir()

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ensure_dir(tmpdir)
            ensure_dir(tmpdir)
            assert Path(tmpdir).is_dir()
