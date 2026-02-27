"""
Tests for the pipeline framework: core Pipeline, steps, YAML loading.
All tests are local — no network calls.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from molfun.pipelines.pipeline import (
    Pipeline,
    PipelineStep,
    StepResult,
    _import_callable,
)


# ── Helpers ──────────────────────────────────────────────────────────

def add_one(state: dict) -> dict:
    return {**state, "x": state.get("x", 0) + 1}


def double(state: dict) -> dict:
    return {**state, "x": state["x"] * 2}


def append_name(state: dict) -> dict:
    trail = state.get("trail", [])
    return {**state, "trail": trail + [state.get("_step_name", "?")]}


def failing_step(state: dict) -> dict:
    raise RuntimeError("Boom!")


def recorder(state: dict) -> dict:
    return {**state, "name": state.get("name", "default")}


# ── Pipeline core ────────────────────────────────────────────────────

class TestPipeline:

    def test_simple_run(self):
        p = Pipeline([
            PipelineStep("add", add_one),
            PipelineStep("double", double),
        ])
        result = p.run({"x": 5})
        assert result["x"] == 12  # (5+1)*2

    def test_empty_state(self):
        p = Pipeline([PipelineStep("add", add_one)])
        result = p.run()
        assert result["x"] == 1

    def test_config_merges_into_state(self):
        p = Pipeline([
            PipelineStep("rec", recorder, config={"name": "alice"}),
        ])
        result = p.run()
        assert result["name"] == "alice"

    def test_config_overrides_state(self):
        p = Pipeline([
            PipelineStep("rec", recorder, config={"name": "bob"}),
        ])
        result = p.run({"name": "alice"})
        assert result["name"] == "bob"

    def test_skip_step(self):
        p = Pipeline([
            PipelineStep("add", add_one),
            PipelineStep("double", double, skip=True),
        ])
        result = p.run({"x": 5})
        assert result["x"] == 6  # only add_one, no doubling

    def test_step_names(self):
        p = Pipeline([
            PipelineStep("a", add_one),
            PipelineStep("b", double),
        ])
        assert p.step_names == ["a", "b"]

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            Pipeline([
                PipelineStep("a", add_one),
                PipelineStep("a", double),
            ])

    def test_dry_run(self):
        p = Pipeline([
            PipelineStep("a", add_one),
            PipelineStep("b", double, skip=True),
            PipelineStep("c", add_one),
        ])
        names = p.dry_run()
        assert names == ["a", "c"]

    def test_describe(self):
        p = Pipeline([
            PipelineStep("a", add_one, config={"k": 1}),
        ])
        desc = p.describe()
        assert desc["type"] == "custom"
        assert len(desc["steps"]) == 1
        assert desc["steps"][0]["name"] == "a"
        assert desc["steps"][0]["config"] == {"k": 1}
        assert "add_one" in desc["steps"][0]["fn"]


# ── Run from / resume ────────────────────────────────────────────────

class TestRunFrom:

    def test_run_from_skips_earlier_steps(self):
        p = Pipeline([
            PipelineStep("a", add_one),
            PipelineStep("b", double),
            PipelineStep("c", add_one),
        ])
        result = p.run_from("b", {"x": 10})
        assert result["x"] == 21  # 10*2 + 1

    def test_run_from_first_step(self):
        p = Pipeline([
            PipelineStep("a", add_one),
            PipelineStep("b", double),
        ])
        result = p.run_from("a", {"x": 0})
        assert result["x"] == 2  # (0+1)*2

    def test_run_from_last_step(self):
        p = Pipeline([
            PipelineStep("a", add_one),
            PipelineStep("b", double),
        ])
        result = p.run_from("b", {"x": 7})
        assert result["x"] == 14

    def test_run_from_unknown_raises(self):
        p = Pipeline([PipelineStep("a", add_one)])
        with pytest.raises(ValueError, match="not found"):
            p.run_from("z", {})


# ── Error handling ───────────────────────────────────────────────────

class TestErrorHandling:

    def test_error_preserves_step_info(self):
        p = Pipeline([
            PipelineStep("ok", add_one),
            PipelineStep("fail", failing_step),
        ])
        with pytest.raises(RuntimeError, match="Boom"):
            p.run({"x": 0})

    def test_error_records_in_state(self):
        p = Pipeline([
            PipelineStep("fail", failing_step),
        ])
        try:
            p.run()
        except RuntimeError:
            pass


# ── Checkpointing ───────────────────────────────────────────────────

class TestCheckpointing:

    def test_checkpoint_creates_files(self, tmp_path):
        p = Pipeline(
            [
                PipelineStep("a", add_one),
                PipelineStep("b", double),
            ],
            checkpoint_dir=str(tmp_path / "ckpts"),
        )
        p.run({"x": 5})
        ckpt_dir = tmp_path / "ckpts"
        assert (ckpt_dir / "state_after_a.json").exists()
        assert (ckpt_dir / "state_after_b.json").exists()

    def test_checkpoint_content(self, tmp_path):
        p = Pipeline(
            [PipelineStep("a", add_one)],
            checkpoint_dir=str(tmp_path / "ckpts"),
        )
        p.run({"x": 5})
        data = json.loads((tmp_path / "ckpts" / "state_after_a.json").read_text())
        assert data["x"] == 6

    def test_non_serializable_values_handled(self, tmp_path):
        def add_object(state):
            return {**state, "obj": object(), "x": 1}

        p = Pipeline(
            [PipelineStep("a", add_object)],
            checkpoint_dir=str(tmp_path / "ckpts"),
        )
        p.run()
        data = json.loads((tmp_path / "ckpts" / "state_after_a.json").read_text())
        assert data["x"] == 1
        assert "object" in data["obj"]


# ── Hooks ────────────────────────────────────────────────────────────

class TestHooks:

    def test_on_step_start_called(self):
        log = []
        hooks = {"on_step_start": lambda name, state: log.append(("start", name))}
        p = Pipeline(
            [PipelineStep("a", add_one), PipelineStep("b", double)],
            hooks=hooks,
        )
        p.run({"x": 0})
        assert log == [("start", "a"), ("start", "b")]

    def test_on_step_end_called(self):
        log = []
        hooks = {"on_step_end": lambda name, state, result: log.append(("end", name, result.elapsed_s >= 0))}
        p = Pipeline(
            [PipelineStep("a", add_one)],
            hooks=hooks,
        )
        p.run({"x": 0})
        assert log == [("end", "a", True)]

    def test_failing_hook_doesnt_crash(self):
        hooks = {"on_step_start": lambda name, state: 1 / 0}
        p = Pipeline(
            [PipelineStep("a", add_one)],
            hooks=hooks,
        )
        result = p.run({"x": 0})
        assert result["x"] == 1


# ── Pipeline results tracking ────────────────────────────────────────

class TestPipelineResults:

    def test_results_in_state(self):
        p = Pipeline([
            PipelineStep("a", add_one),
            PipelineStep("b", double),
        ])
        result = p.run({"x": 0})
        results = result["_pipeline_results"]
        assert len(results) == 2
        assert all(isinstance(r, StepResult) for r in results)
        assert results[0].name == "a"
        assert results[1].name == "b"
        assert all(r.elapsed_s >= 0 for r in results)
        assert all(not r.skipped for r in results)

    def test_skipped_in_results(self):
        p = Pipeline([
            PipelineStep("a", add_one),
            PipelineStep("b", double, skip=True),
        ])
        result = p.run({"x": 0})
        results = result["_pipeline_results"]
        assert results[1].skipped is True


# ── YAML loading ─────────────────────────────────────────────────────

class TestYAML:

    def test_from_yaml(self, tmp_path):
        recipe = tmp_path / "test.yaml"
        recipe.write_text("""
checkpoint_dir: /tmp/test_ckpt

steps:
  - name: add
    fn: tests.pipelines.test_pipeline.add_one
    config:
      x: 5
  - name: double
    fn: tests.pipelines.test_pipeline.double
""")
        p = Pipeline.from_yaml(str(recipe))
        assert len(p.steps) == 2
        assert p.steps[0].name == "add"
        assert p.checkpoint_dir == "/tmp/test_ckpt"

        result = p.run()
        assert result["x"] == 12

    def test_from_yaml_with_overrides(self, tmp_path):
        recipe = tmp_path / "test.yaml"
        recipe.write_text("""
steps:
  - name: rec
    fn: tests.pipelines.test_pipeline.recorder
    config:
      name: original
""")
        p = Pipeline.from_yaml(str(recipe), name="override")
        result = p.run()
        assert result["name"] == "override"

    def test_from_yaml_skip(self, tmp_path):
        recipe = tmp_path / "test.yaml"
        recipe.write_text("""
steps:
  - name: add
    fn: tests.pipelines.test_pipeline.add_one
  - name: double
    fn: tests.pipelines.test_pipeline.double
    skip: true
""")
        p = Pipeline.from_yaml(str(recipe))
        result = p.run({"x": 5})
        assert result["x"] == 6


# ── Import callable ──────────────────────────────────────────────────

class TestImportCallable:

    def test_import_builtin(self):
        fn = _import_callable("json.dumps")
        assert callable(fn)

    def test_import_our_step(self):
        fn = _import_callable("tests.pipelines.test_pipeline.add_one")
        assert fn is add_one

    def test_invalid_path_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            _import_callable("no_dots")

    def test_missing_function_raises(self):
        with pytest.raises(ImportError, match="Cannot find"):
            _import_callable("json.nonexistent_function_xyz")


# ── Built-in steps (unit tests) ─────────────────────────────────────

class TestBuiltInSteps:

    def test_split_step(self):
        from molfun.pipelines.steps import split_step

        paths = [f"/data/{i}.cif" for i in range(100)]
        result = split_step({
            "pdb_paths": paths,
            "val_frac": 0.1,
            "test_frac": 0.1,
            "seed": 42,
        })
        assert result["n_train"] + result["n_val"] + result["n_test"] == 100
        assert result["n_val"] >= 1
        assert result["n_test"] >= 1
        assert len(set(result["train_paths"]) & set(result["val_paths"])) == 0
        assert len(set(result["train_paths"]) & set(result["test_paths"])) == 0

    def test_split_step_deterministic(self):
        from molfun.pipelines.steps import split_step

        paths = [f"/data/{i}.cif" for i in range(50)]
        r1 = split_step({"pdb_paths": paths, "seed": 42})
        r2 = split_step({"pdb_paths": paths, "seed": 42})
        assert r1["train_paths"] == r2["train_paths"]

    @patch("molfun.data.sources.pdb.PDBFetcher")
    def test_fetch_step_with_collection(self, mock_cls):
        from molfun.pipelines.steps import fetch_step

        mock_fetcher = MagicMock()
        mock_cls.return_value = mock_fetcher
        with patch("molfun.data.collections.PDBFetcher", mock_cls):
            with patch("molfun.data.collections.fetch_collection", return_value=["/a.cif", "/b.cif"]):
                result = fetch_step({"collection": "kinases", "output_dir": "/tmp"})
                assert result["n_structures"] == 2
                assert result["pdb_paths"] == ["/a.cif", "/b.cif"]

    @patch("molfun.data.sources.pdb.PDBFetcher")
    def test_fetch_step_with_filters(self, mock_cls):
        from molfun.pipelines.steps import fetch_step

        mock_fetcher = MagicMock()
        mock_cls.return_value = mock_fetcher
        mock_fetcher.search_ids.return_value = ["1abc", "2xyz"]
        mock_fetcher.fetch.return_value = ["/1abc.cif", "/2xyz.cif"]

        result = fetch_step({"pfam_id": "PF00069", "output_dir": "/tmp"})
        assert result["n_structures"] == 2

    def test_save_step(self, tmp_path):
        from molfun.pipelines.steps import save_step

        mock_model = MagicMock()
        ckpt_dir = str(tmp_path / "ckpt")
        result = save_step({"model": mock_model, "checkpoint_dir": ckpt_dir})
        mock_model.save.assert_called_once_with(ckpt_dir)
        assert result["checkpoint_path"] == ckpt_dir

    def test_push_step_requires_repo(self):
        from molfun.pipelines.steps import push_step
        with pytest.raises(ValueError, match="repo"):
            push_step({"checkpoint_path": "/some/path"})

    def test_push_step_requires_checkpoint(self):
        from molfun.pipelines.steps import push_step
        with pytest.raises(ValueError, match="checkpoint_path"):
            push_step({"repo": "user/model"})
