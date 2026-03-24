"""
Configuration — Centralized settings for the cognitive memory system.

All user-facing knobs live here. Pass a Config instance to MemorySystem
instead of scattering keyword arguments across constructors.

Every field can be overridden via environment variable using the
``COGNITIVE_MEMORY_`` prefix (e.g. ``COGNITIVE_MEMORY_MODEL``).
Call ``Config.from_env()`` to build a Config from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


@dataclass
class Config:
    """Centralized configuration for the cognitive memory system."""

    # ── Paths ─────────────────────────────────────────────────────────────
    genome_path: str = "genome.json"
    graph_path: str = "graph.json"
    entities_path: str = "entities.json"

    # ── Model ─────────────────────────────────────────────────────────────
    model: str = "claude-sonnet-4-20250514"

    # ── Encoding ──────────────────────────────────────────────────────────
    max_extraction_retries: int = 2
    max_entities_per_strand: int = 4

    # ── Graph ─────────────────────────────────────────────────────────────
    decay_factor: float = 0.99
    decay_floor: float = 0.01
    initial_temporal_weight: float = 0.6
    initial_entity_weight: float = 0.8
    initial_semantic_weight: float = 0.5
    hebbian_increment: float = 0.15
    ego_edge_weight: float = 0.9
    max_edges_per_node: int = 50
    recency_bonus: float = 0.3
    recency_decay_rate: float = 0.85

    # ── Expression (retrieval) ────────────────────────────────────────────
    base_threshold: float = 0.15
    spread_decay: float = 0.7
    token_budget: int = 500
    seed_count: int = 3
    max_spread_depth: int = 4

    # ── Forgetting defaults ───────────────────────────────────────────────
    forget_min_age_seconds: int = 86400 * 30
    forget_min_activations: int = 0

    # ── Environment variable support ─────────────────────────────────────

    _ENV_PREFIX = "COGNITIVE_MEMORY_"

    @classmethod
    def from_env(cls, **overrides) -> Config:
        """Build a Config, applying environment variable overrides.

        Environment variables use the ``COGNITIVE_MEMORY_`` prefix with
        the uppercased field name. For example::

            COGNITIVE_MEMORY_MODEL=claude-haiku-4-5-20251001
            COGNITIVE_MEMORY_TOKEN_BUDGET=800
            COGNITIVE_MEMORY_GENOME_PATH=/data/genome.json

        Explicit ``overrides`` kwargs take precedence over env vars,
        which take precedence over dataclass defaults.
        """
        kwargs: dict = {}
        for f in fields(cls):
            if f.name.startswith("_"):
                continue
            env_key = f"{cls._ENV_PREFIX}{f.name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                kwargs[f.name] = _coerce(env_val, f.type)
        kwargs.update(overrides)
        return cls(**kwargs)


def _coerce(value: str, type_hint: str) -> int | float | str:
    """Coerce a string env var value to the appropriate Python type."""
    if type_hint == "int":
        return int(value)
    if type_hint == "float":
        return float(value)
    return value
