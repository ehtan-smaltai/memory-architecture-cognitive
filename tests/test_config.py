"""Tests for Config — defaults, env var overrides, and coercion."""

import os
import pytest
from cognitive_memory import Config


class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.model == "claude-sonnet-4-20250514"
        assert cfg.token_budget == 2000
        assert cfg.decay_factor == 0.99

    def test_explicit_override(self):
        cfg = Config(model="custom-model", token_budget=800)
        assert cfg.model == "custom-model"
        assert cfg.token_budget == 800

    def test_from_env_string(self, monkeypatch):
        monkeypatch.setenv("COGNITIVE_MEMORY_MODEL", "claude-haiku-4-5-20251001")
        cfg = Config.from_env()
        assert cfg.model == "claude-haiku-4-5-20251001"

    def test_from_env_int(self, monkeypatch):
        monkeypatch.setenv("COGNITIVE_MEMORY_TOKEN_BUDGET", "1000")
        cfg = Config.from_env()
        assert cfg.token_budget == 1000

    def test_from_env_float(self, monkeypatch):
        monkeypatch.setenv("COGNITIVE_MEMORY_DECAY_FACTOR", "0.95")
        cfg = Config.from_env()
        assert cfg.decay_factor == 0.95

    def test_from_env_explicit_overrides_env(self, monkeypatch):
        """Explicit kwargs beat env vars."""
        monkeypatch.setenv("COGNITIVE_MEMORY_MODEL", "env-model")
        cfg = Config.from_env(model="explicit-model")
        assert cfg.model == "explicit-model"

    def test_from_env_ignores_unset(self):
        cfg = Config.from_env()
        assert cfg.model == "claude-sonnet-4-20250514"  # default


class TestConfigPaths:
    def test_default_paths(self):
        cfg = Config()
        assert cfg.genome_path == "genome.json"
        assert cfg.graph_path == "graph.json"
        assert cfg.entities_path == "entities.json"

    def test_custom_paths(self):
        cfg = Config(genome_path="/tmp/g.json", graph_path="/tmp/gr.json")
        assert cfg.genome_path == "/tmp/g.json"
