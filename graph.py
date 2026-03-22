"""
Association Graph — Layer 2 of the Cognitive Memory Architecture

A weighted directed graph where nodes are strand_ids and edges represent
association strength. Implements Hebbian learning (co-activation strengthens
connections) and temporal decay.

Edge types:
  - temporal:      strands close in time
  - entity_shared: strands sharing named entities
  - semantic:      strands in the same domain with related values
  - causal:        co-activated during expression (Hebbian)
"""

import json
import os
import time

import networkx as nx


class AssociationGraph:
    """Weighted directed graph over memory strands."""

    DECAY_FACTOR = 0.99          # per-retrieval-cycle weight decay
    INITIAL_TEMPORAL_WEIGHT = 0.6
    INITIAL_ENTITY_WEIGHT = 0.8
    INITIAL_SEMANTIC_WEIGHT = 0.5
    HEBBIAN_INCREMENT = 0.15     # co-activation boost

    def __init__(self, path: str = "graph.json"):
        self.path = path
        self.graph = nx.DiGraph()
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
            for node in data.get("nodes", []):
                self.graph.add_node(node)
            for edge in data.get("edges", []):
                self.graph.add_edge(
                    edge["source"],
                    edge["target"],
                    weight=edge["weight"],
                    edge_type=edge["edge_type"],
                    created=edge.get("created", 0),
                )

    def save(self):
        data = {
            "nodes": list(self.graph.nodes),
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "weight": d["weight"],
                    "edge_type": d["edge_type"],
                    "created": d.get("created", 0),
                }
                for u, v, d in self.graph.edges(data=True)
            ],
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Node / edge operations ───────────────────────────────────────────

    def add_strand(self, strand: dict, recent_ids: list[str], all_strands_getter=None):
        """
        Register a new strand in the graph and auto-create initial edges.

        Args:
            strand: the encoded strand dict
            recent_ids: the 3 most recently added strand IDs (for temporal edges)
            all_strands_getter: callable(strand_id) -> strand dict, for entity checks
        """
        sid = strand["strand_id"]
        self.graph.add_node(sid)

        # Temporal edges to recent strands
        for rid in recent_ids[-3:]:
            if rid != sid and self.graph.has_node(rid):
                self._add_edge(sid, rid, self.INITIAL_TEMPORAL_WEIGHT, "temporal")
                self._add_edge(rid, sid, self.INITIAL_TEMPORAL_WEIGHT, "temporal")

        # Entity-shared edges
        if all_strands_getter:
            my_entities = set(strand["encoded"]["entities"])
            for nid in list(self.graph.nodes):
                if nid == sid:
                    continue
                other = all_strands_getter(nid)
                if other is None:
                    continue
                other_entities = set(other["encoded"]["entities"])
                shared = my_entities & other_entities
                if len(shared) >= 1:
                    weight = min(1.0, self.INITIAL_ENTITY_WEIGHT * len(shared))
                    self._add_edge(sid, nid, weight, "entity_shared")
                    self._add_edge(nid, sid, weight, "entity_shared")

                # Semantic edges — same domain + related value
                if (
                    strand["encoded"]["domain"] == other["encoded"]["domain"]
                    and strand["encoded"]["value"] == other["encoded"]["value"]
                ):
                    self._add_edge(sid, nid, self.INITIAL_SEMANTIC_WEIGHT, "semantic")
                    self._add_edge(nid, sid, self.INITIAL_SEMANTIC_WEIGHT, "semantic")

        self.save()

    def _add_edge(self, src: str, tgt: str, weight: float, edge_type: str):
        """Add or strengthen an edge."""
        if self.graph.has_edge(src, tgt):
            existing = self.graph[src][tgt]
            # Keep the stronger weight
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

    # ── Hebbian learning ─────────────────────────────────────────────────

    def hebbian_update(self, co_activated_ids: list[str]):
        """
        Strengthen edges between all strands that were co-activated in a
        single expression event. Creates 'causal' edges if none exist.
        """
        for i, a in enumerate(co_activated_ids):
            for b in co_activated_ids[i + 1:]:
                if self.graph.has_edge(a, b):
                    self.graph[a][b]["weight"] = min(
                        1.0, self.graph[a][b]["weight"] + self.HEBBIAN_INCREMENT
                    )
                else:
                    self._add_edge(a, b, self.HEBBIAN_INCREMENT, "causal")
                if self.graph.has_edge(b, a):
                    self.graph[b][a]["weight"] = min(
                        1.0, self.graph[b][a]["weight"] + self.HEBBIAN_INCREMENT
                    )
                else:
                    self._add_edge(b, a, self.HEBBIAN_INCREMENT, "causal")
        self.save()

    # ── Decay ────────────────────────────────────────────────────────────

    def apply_decay(self):
        """Apply temporal decay to all edge weights."""
        for u, v, d in self.graph.edges(data=True):
            d["weight"] *= self.DECAY_FACTOR
        self.save()

    # ── Query helpers ────────────────────────────────────────────────────

    def neighbors(self, strand_id: str) -> list[tuple[str, float, str]]:
        """Return [(neighbor_id, weight, edge_type), ...] for a node."""
        if not self.graph.has_node(strand_id):
            return []
        result = []
        for _, tgt, d in self.graph.edges(strand_id, data=True):
            result.append((tgt, d["weight"], d["edge_type"]))
        return result

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()
