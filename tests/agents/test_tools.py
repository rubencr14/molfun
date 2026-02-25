"""Tests for MolfunTools (tool schemas and dispatch)."""

import json
import pytest

from molfun.agents.tools import MolfunTools, TOOL_SCHEMAS


class TestToolSchemas:
    def test_all_schemas_have_required_fields(self):
        for schema in TOOL_SCHEMAS:
            assert "type" in schema
            assert schema["type"] == "function"
            fn = schema["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    def test_schema_names_unique(self):
        names = [s["function"]["name"] for s in TOOL_SCHEMAS]
        assert len(names) == len(set(names))


class TestMolfunTools:
    @pytest.fixture
    def tools(self):
        """Tools with None loaders (only test dispatch, not actual training)."""
        return MolfunTools(
            train_loader=None,
            val_loader=None,
            device="cpu",
        )

    def test_schemas_property(self, tools):
        schemas = tools.schemas
        assert len(schemas) > 0
        assert all(s["type"] == "function" for s in schemas)

    def test_list_components(self, tools):
        result = tools.execute("list_components", {})
        data = json.loads(result)
        assert "attention" in data
        assert "blocks" in data
        assert "structure_modules" in data
        assert "embedders" in data
        assert "strategies" in data

        assert "standard" in data["attention"]
        assert "pairformer" in data["blocks"]
        assert "ipa" in data["structure_modules"]

    def test_unknown_tool(self, tools):
        result = tools.execute("nonexistent_tool", {})
        assert "Error" in result
        assert "nonexistent_tool" in result

    def test_done(self, tools):
        assert not tools.is_done
        result = tools.execute("done", {"summary": "Test done"})
        assert tools.is_done
        assert tools.done_summary == "Test done"

    def test_get_journal_placeholder(self, tools):
        result = tools.execute("get_journal", {})
        assert "memory" in result.lower()

    def test_save_best_no_model(self, tools):
        result = tools.execute("save_best_model", {"path": "/tmp/test"})
        data = json.loads(result)
        assert "error" in data
