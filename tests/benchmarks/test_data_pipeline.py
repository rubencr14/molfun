"""Tests for molfun.benchmarks.data_pipeline."""

import tempfile
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from molfun.benchmarks.data_pipeline import (
    ParsingBenchmark,
    ParsingReport,
    LoadingBenchmark,
    LoadingReport,
)


class TestParsingBenchmark:
    def test_missing_file(self):
        bench = ParsingBenchmark(repeats=2, warmup=0)
        report = bench.run(["/nonexistent/file.pdb"])
        assert len(report.results) == 1
        assert not report.results[0].success
        assert "not found" in report.results[0].error.lower()

    def test_unknown_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz") as f:
            bench = ParsingBenchmark(repeats=2, warmup=0)
            report = bench.run([f.name])
            assert len(report.results) == 1
            assert not report.results[0].success

    def test_valid_pdb(self, tmp_path):
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
            "ATOM      2  CA  GLY A   2       4.000   5.000   6.000  1.00  0.00           C\n"
            "END\n"
        )
        bench = ParsingBenchmark(repeats=3, warmup=1)
        report = bench.run([str(pdb_file)])
        assert len(report.results) == 1
        assert report.results[0].success
        assert report.results[0].parser_name == "PDBParser"
        assert report.results[0].throughput_files_per_s > 0

    def test_mean_throughput(self, tmp_path):
        pdb = tmp_path / "t.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\nEND\n")
        bench = ParsingBenchmark(repeats=2, warmup=0)
        report = bench.run([str(pdb)])
        assert report.mean_throughput() > 0

    def test_report_markdown(self, tmp_path):
        pdb = tmp_path / "t.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\nEND\n")
        bench = ParsingBenchmark(repeats=2, warmup=0)
        report = bench.run([str(pdb)])
        md = report.to_markdown()
        assert "PDBParser" in md
        assert "yes" in md


class TestLoadingBenchmark:
    def test_basic_throughput(self):
        ds = TensorDataset(torch.randn(20, 4), torch.randn(20))
        bench = LoadingBenchmark(ds, max_batches=5)
        report = bench.run(worker_counts=[0], batch_sizes=[4])
        assert len(report.results) == 1
        r = report.results[0]
        assert r.num_workers == 0
        assert r.batch_size == 4
        assert r.samples_per_s > 0
        assert r.total_samples > 0

    def test_multiple_configs(self):
        ds = TensorDataset(torch.randn(20, 4))
        bench = LoadingBenchmark(ds, max_batches=3)
        report = bench.run(worker_counts=[0], batch_sizes=[1, 4])
        assert len(report.results) == 2

    def test_report_markdown(self):
        ds = TensorDataset(torch.randn(10, 2))
        bench = LoadingBenchmark(ds, max_batches=2)
        report = bench.run(worker_counts=[0], batch_sizes=[2])
        md = report.to_markdown()
        assert "Samples/s" in md
