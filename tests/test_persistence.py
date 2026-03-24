"""Tests for versioned persistence — save/load with schema version."""

import json
import os
import tempfile
import pytest
from cognitive_memory._persistence import (
    SCHEMA_VERSION, save_versioned, load_versioned, atomic_write_json,
)


class TestVersionedPersistence:
    def setup_method(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmpfile.close()
        os.unlink(self.tmpfile.name)

    def teardown_method(self):
        if os.path.exists(self.tmpfile.name):
            os.unlink(self.tmpfile.name)

    def test_save_and_load_roundtrip(self):
        items = [{"id": "a"}, {"id": "b"}]
        save_versioned(self.tmpfile.name, "strands", items)

        version, loaded, extra = load_versioned(self.tmpfile.name, "strands")
        assert version == SCHEMA_VERSION
        assert len(loaded) == 2
        assert loaded[0]["id"] == "a"

    def test_load_nonexistent_returns_empty(self):
        version, items, extra = load_versioned("/nonexistent/path.json", "strands")
        assert version == SCHEMA_VERSION
        assert items == []
        assert extra == {}

    def test_load_legacy_bare_list(self):
        """Pre-versioning files are bare JSON lists — should still load."""
        legacy_data = [{"strand_id": "s1"}, {"strand_id": "s2"}]
        with open(self.tmpfile.name, "w") as f:
            json.dump(legacy_data, f)

        version, items, extra = load_versioned(self.tmpfile.name, "strands")
        assert version == 0  # legacy
        assert len(items) == 2

    def test_extra_fields_preserved(self):
        save_versioned(self.tmpfile.name, "nodes", [{"id": "n1"}],
                       edges=[{"src": "a", "tgt": "b"}],
                       recency_buffer=[])

        version, nodes, extra = load_versioned(self.tmpfile.name, "nodes")
        assert version == SCHEMA_VERSION
        assert len(nodes) == 1
        assert "edges" in extra
        assert len(extra["edges"]) == 1

    def test_version_field_in_file(self):
        save_versioned(self.tmpfile.name, "items", [])
        with open(self.tmpfile.name) as f:
            data = json.load(f)
        assert "version" in data
        assert data["version"] == SCHEMA_VERSION


class TestAtomicWrite:
    def test_atomic_write_creates_file(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        os.unlink(tmp.name)
        try:
            atomic_write_json(tmp.name, {"key": "value"})
            with open(tmp.name) as f:
                data = json.load(f)
            assert data["key"] == "value"
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
