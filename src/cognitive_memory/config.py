"""
Configuration — Centralized settings for the cognitive memory system.

All user-facing knobs live here. Pass a Config instance to MemorySystem
instead of scattering keyword arguments across constructors.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
