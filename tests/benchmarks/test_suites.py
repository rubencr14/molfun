"""Tests for molfun.benchmarks.suites."""

from molfun.benchmarks.suites import BenchmarkTask, BenchmarkSuite, TaskType


class TestBenchmarkTask:
    def test_immutable(self):
        task = BenchmarkTask(name="test", data_source="/data")
        try:
            task.name = "other"
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_defaults(self):
        task = BenchmarkTask(name="t", data_source="/d")
        assert task.split == "test"
        assert task.task_type == TaskType.REGRESSION
        assert task.max_samples is None


class TestBenchmarkSuite:
    def test_pdbbind(self):
        suite = BenchmarkSuite.pdbbind()
        assert suite.name == "PDBbind-v2020"
        assert len(suite) == 2
        assert "mae" in suite.tasks[0].metrics

    def test_atom3d_lba(self):
        suite = BenchmarkSuite.atom3d_lba()
        assert len(suite) >= 1
        assert suite.tasks[0].task_type == TaskType.REGRESSION

    def test_flip(self):
        suite = BenchmarkSuite.flip()
        assert len(suite) == 3
        names = suite.task_names()
        assert "flip_aav" in names
        assert "flip_gb1" in names

    def test_structure_quality(self):
        suite = BenchmarkSuite.structure_quality()
        assert suite.tasks[0].task_type == TaskType.STRUCTURE
        assert "gdt_ts" in suite.tasks[0].metrics

    def test_custom(self):
        tasks = [
            BenchmarkTask(name="my_task", data_source="/data", metrics=("mae",)),
        ]
        suite = BenchmarkSuite.custom("MySuite", tasks, description="test")
        assert suite.name == "MySuite"
        assert len(suite) == 1

    def test_summary(self):
        suite = BenchmarkSuite.pdbbind()
        s = suite.summary()
        assert "PDBbind" in s
        assert "mae" in s

    def test_iterable(self):
        suite = BenchmarkSuite.pdbbind()
        tasks = list(suite)
        assert len(tasks) == 2
