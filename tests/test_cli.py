"""Tests for the CLI entry point."""

import os
import tempfile
import pytest
from cognitive_memory.__main__ import main


class TestCLI:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.genome = os.path.join(self.tmpdir, "genome.json")
        self.graph = os.path.join(self.tmpdir, "graph.json")
        self.entities = os.path.join(self.tmpdir, "entities.json")
        self.base_args = [
            "--genome", self.genome,
            "--graph", self.graph,
            "--entities", self.entities,
        ]

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_help_returns_zero(self):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0

    def test_no_command_returns_zero(self):
        assert main([]) == 0

    def test_stats_empty(self):
        result = main(self.base_args + ["stats"])
        assert result == 0

    def test_export_empty(self, capsys):
        result = main(self.base_args + ["export"])
        assert result == 0
        captured = capsys.readouterr()
        assert "[]" in captured.out

    def test_consolidate_empty(self):
        result = main(self.base_args + ["consolidate"])
        assert result == 0

    def test_forget_empty(self):
        result = main(self.base_args + ["forget"])
        assert result == 0
