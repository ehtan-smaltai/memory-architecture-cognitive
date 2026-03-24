"""
Association Graph — Layer 2 of the Cognitive Memory Architecture (v2)

A weighted directed graph where nodes are strand_ids (or ego nodes) and
edges represent association strength. Implements:
  - Hebbian learning (co-activation → stronger connections)
  - Temporal decay
  - Ego nodes (agent identity / personally significant memories)
  - Recency priming (recently activated paths stay "warm")
  - Entity-registry-based edges (accurate cross-strand entity linking)

Edge types:
  temporal      — strands close in time
  entity_shared — strands sharing an entity instance (via registry)
  semantic      — strands in the same domain with related codes
  causal        — co-activated during expression (Hebbian)
  ego           — links from ego nodes to significant strands
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import networkx as nx

from codebook import CodebookStrand
from entities import EntityRegistry


def _atomic_write_json(path: str, data) -> None:
    """Write JSON atomically: write to temp file, then os.replace."""
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class AssociationGraph:
    """Weighted directed graph over memory strands + ego nodes."""

    # Edge weights
    DECAY_FACTOR = 0.99
    DECAY_FLOOR = 0.01  # minimum edge weight — prevents decay from killing edges
    INITIAL_TEMPORAL_WEIGHT = 0.6
    INITIAL_ENTITY_WEIGHT = 0.8
    INITIAL_SEMANTIC_WEIGHT = 0.5
    HEBBIAN_INCREMENT = 0.15
    EGO_EDGE_WEIGHT = 0.9
    MAX_EDGES_PER_NODE = 50  # prevent edge explosion for popular entities

    # Recency priming
    RECENCY_BONUS = 0.3
    RECENCY_DECAY_RATE = 0.85

    # Ego node prefix
    EGO_NODE_PREFIX = "ego:"

    def __init__(self, path: str = "graph.json"):
        self.path = path
        self.graph = nx.DiGraph()
        self._recency_buffer: dict[str, float] = {}  # strand_id → warmth
        # Index: (domain, relation) → set of strand_ids for O(1) semantic lookups
        self._domain_relation_index: dict[tuple[int, int], set[str]] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
            for node in data.get("nodes", []):
                if isinstance(node, dict):
                    self.graph.add_node(node["id"], node_type=node.get("node_type", "strand"))
                else:
                    self.graph.add_node(node, node_type="strand")
            for edge in data.get("edges", []):
                self.graph.add_edge(
                    edge["source"],
                    edge["target"],
                    weight=edge["weight"],
                    edge_type=edge["edge_type"],
                    created=edge.get("created", 0),
                )
            # Restore recency buffer if present
            for entry in data.get("recency_buffer", []):
                self._recency_buffer[entry["id"]] = entry["warmth"]

    def rebuild_domain_relation_index(self, genome_getter):
        """Rebuild the domain+relation index from the genome (call after load)."""
        self._domain_relation_index.clear()
        for nid in self.graph.nodes:
            if str(nid).startswith(self.EGO_NODE_PREFIX):
                continue
            strand = genome_getter(nid)
            if strand is None:
                continue
            key = (strand.domain, strand.relation)
            if key not in self._domain_relation_index:
                self._domain_relation_index[key] = set()
            self._domain_relation_index[key].add(nid)

    def save(self):
        nodes = []
        for n in self.graph.nodes:
            node_type = self.graph.nodes[n].get("node_type", "strand")
            nodes.append({"id": n, "node_type": node_type})

        edges = [
            {
                "source": u,
                "target": v,
                "weight": d["weight"],
                "edge_type": d["edge_type"],
                "created": d.get("created", 0),
            }
            for u, v, d in self.graph.edges(data=True)
        ]

        recency = [
            {"id": sid, "warmth": w}
            for sid, w in self._recency_buffer.items()
        ]

        data = {"nodes": nodes, "edges": edges, "recency_buffer": recency}
        _atomic_write_json(self.path, data)

    # ── Ego nodes ────────────────────────────────────────────────────────

    def ensure_ego_node(self, ego_id: str = "agent"):
        """Create the ego node if it doesn't exist."""
        full_id = f"{self.EGO_NODE_PREFIX}{ego_id}"
        if not self.graph.has_node(full_id):
            self.graph.add_node(full_id, node_type="ego")
            self.save()

    def link_to_ego(self, strand_id: str, ego_id: str = "agent"):
        """Create a high-weight edge between an ego node and a strand."""
        full_id = f"{self.EGO_NODE_PREFIX}{ego_id}"
        self.ensure_ego_node(ego_id)
        if self.graph.has_node(strand_id):
            self._add_edge(full_id, strand_id, self.EGO_EDGE_WEIGHT, "ego")
            self._add_edge(strand_id, full_id, self.EGO_EDGE_WEIGHT * 0.5, "ego")
            self.save()

    def get_ego_linked_strands(self, ego_id: str = "agent") -> list[str]:
        """Return all strand_ids linked to an ego node."""
        full_id = f"{self.EGO_NODE_PREFIX}{ego_id}"
        if not self.graph.has_node(full_id):
            return []
        return [
            tgt for _, tgt, d in self.graph.edges(full_id, data=True)
            if d["edge_type"] == "ego"
        ]

    # ── Strand registration ──────────────────────────────────────────────

    def add_strand(
        self,
        strand: CodebookStrand,
        recent_ids: list[str],
        entity_registry: EntityRegistry | None = None,
        genome_getter=None,
    ):
        """
        Register a new strand in the graph and auto-create edges.

        Uses EntityRegistry for accurate entity-based edges instead of
        raw string matching.
        """
        sid = strand.strand_id
        self.graph.add_node(sid, node_type="strand")

        # Temporal edges to 3 most recent strands
        for rid in recent_ids[-3:]:
            if rid != sid and self.graph.has_node(rid):
                self._add_edge(sid, rid, self.INITIAL_TEMPORAL_WEIGHT, "temporal")
                self._add_edge(rid, sid, self.INITIAL_TEMPORAL_WEIGHT, "temporal")

        # Entity-shared edges via registry
        if entity_registry is not None:
            my_instance_ids = set(strand.get_entity_instance_ids())
            connected_strands = set()
            for inst_id in my_instance_ids:
                for other_sid in entity_registry.get_strands_for_entity(inst_id):
                    if other_sid != sid and other_sid not in connected_strands:
                        connected_strands.add(other_sid)

            for other_sid in connected_strands:
                if self.graph.has_node(other_sid):
                    # Weight scales with number of shared entities
                    if genome_getter:
                        other = genome_getter(other_sid)
                        if other:
                            shared = my_instance_ids & set(other.get_entity_instance_ids())
                            weight = min(1.0, self.INITIAL_ENTITY_WEIGHT * len(shared))
                        else:
                            weight = self.INITIAL_ENTITY_WEIGHT
                    else:
                        weight = self.INITIAL_ENTITY_WEIGHT
                    self._add_edge(sid, other_sid, weight, "entity_shared")
                    self._add_edge(other_sid, sid, weight, "entity_shared")

        # Semantic edges — same domain + relation code (O(1) via index)
        key = (strand.domain, strand.relation)
        for nid in self._domain_relation_index.get(key, set()):
            if nid != sid and self.graph.has_node(nid):
                self._add_edge(sid, nid, self.INITIAL_SEMANTIC_WEIGHT, "semantic")
                self._add_edge(nid, sid, self.INITIAL_SEMANTIC_WEIGHT, "semantic")

        # Update the index with this strand
        if key not in self._domain_relation_index:
            self._domain_relation_index[key] = set()
        self._domain_relation_index[key].add(sid)

        self.save()

    def _add_edge(self, src: str, tgt: str, weight: float, edge_type: str):
        """Add or strengthen an edge. Prunes weakest edges if cap exceeded."""
        if self.graph.has_edge(src, tgt):
            existing = self.graph[src][tgt]
            if weight > existing["weight"]:
                existing["weight"] = weight
                existing["edge_type"] = edge_type
        else:
            self.graph.add_edge(
                src, tgt,
                weight=weight,
                edge_type=edge_type,
                created=int(time.time()),
            )
            # Prune weakest edges if node exceeds cap
            out_degree = self.graph.out_degree(src)
            if out_degree > self.MAX_EDGES_PER_NODE:
                edges = [
                    (s, t, d) for s, t, d in self.graph.edges(src, data=True)
                    if d["edge_type"] != "ego"  # never prune ego edges
                ]
                edges.sort(key=lambda e: e[2]["weight"])
                # Remove the weakest edges to get back to cap
                for s, t, d in edges[: out_degree - self.MAX_EDGES_PER_NODE]:
                    self.graph.remove_edge(s, t)

    # ── Hebbian learning ─────────────────────────────────────────────────

    def hebbian_update(self, co_activated_ids: list[str]):
        """Strengthen edges between co-activated strands."""
        for i, a in enumerate(co_activated_ids):
            for b in co_activated_ids[i + 1:]:
                for src, tgt in [(a, b), (b, a)]:
                    if self.graph.has_edge(src, tgt):
                        self.graph[src][tgt]["weight"] = min(
                            1.0, self.graph[src][tgt]["weight"] + self.HEBBIAN_INCREMENT
                        )
                    else:
                        self._add_edge(src, tgt, self.HEBBIAN_INCREMENT, "causal")
        self.save()

    # ── Decay ────────────────────────────────────────────────────────────

    def apply_decay(self):
        """Apply temporal decay to all edge weights. Ego edges are exempt. Floor prevents zeroing."""
        for u, v, d in self.graph.edges(data=True):
            if d["edge_type"] == "ego":
                continue  # ego edges don't decay
            d["weight"] = max(self.DECAY_FLOOR, d["weight"] * self.DECAY_FACTOR)
        self.save()

    # ── Recency priming ──────────────────────────────────────────────────

    def prime_recency(self, strand_ids: list[str]):
        """Mark strands as recently activated — they get a warmth bonus."""
        for sid in strand_ids:
            self._recency_buffer[sid] = self.RECENCY_BONUS

    def get_recency_bonus(self, strand_id: str) -> float:
        """Get the current recency warmth for a strand."""
        return self._recency_buffer.get(strand_id, 0.0)

    def decay_recency(self):
        """Decay all recency bonuses. Called each retrieval cycle."""
        to_remove = []
        for sid in self._recency_buffer:
            self._recency_buffer[sid] *= self.RECENCY_DECAY_RATE
            if self._recency_buffer[sid] < 0.05:
                to_remove.append(sid)
        for sid in to_remove:
            del self._recency_buffer[sid]

    # ── Query helpers ────────────────────────────────────────────────────

    def neighbors(self, strand_id: str) -> list[tuple[str, float, str]]:
        """Return [(neighbor_id, weight, edge_type), ...] for a node."""
        if not self.graph.has_node(strand_id):
            return []
        return [
            (tgt, d["weight"], d["edge_type"])
            for _, tgt, d in self.graph.edges(strand_id, data=True)
        ]

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def ego_node_count(self) -> int:
        return sum(
            1 for n in self.graph.nodes
            if str(n).startswith(self.EGO_NODE_PREFIX)
        )
