"""Tests for the association graph — edges, Hebbian learning, decay, ego nodes."""

import os
import tempfile
import pytest
from codebook import (
    CodebookStrand, EntityType, RelationType, Modifier, TemporalMarker, Domain,
    make_codebook_strand,
)
from entities import EntityRegistry
from graph import AssociationGraph


def _make_strand(strand_id, entity_ids=None, domain=0, relation=0, raw_hash=None):
    if entity_ids is None:
        entity_ids = ["alice"]
    strand = make_codebook_strand(
        entity_slots=[(EntityType.PERSON.value, eid) for eid in entity_ids],
        relation=relation,
        modifier=Modifier.NEUTRAL.value,
        temporal=TemporalMarker.PRESENT.value,
        domain=domain,
        sentiment=0,
        confidence=3,
        timestamp=1000,
        raw_hash=raw_hash or strand_id,
    )
    strand.strand_id = strand_id
    return strand


class TestAssociationGraph:
    def setup_method(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmpfile.close()
        os.unlink(self.tmpfile.name)
        self.graph = AssociationGraph(path=self.tmpfile.name)
        self._strands = {}

    def teardown_method(self):
        if os.path.exists(self.tmpfile.name):
            os.unlink(self.tmpfile.name)

    def _genome_getter(self, sid):
        return self._strands.get(sid)

    def test_add_strand_creates_node(self):
        s = _make_strand("s1")
        self._strands["s1"] = s
        self.graph.add_strand(s, recent_ids=[], genome_getter=self._genome_getter)
        assert self.graph.node_count() == 1

    def test_temporal_edges(self):
        s1 = _make_strand("s1")
        s2 = _make_strand("s2")
        self._strands.update({"s1": s1, "s2": s2})

        self.graph.add_strand(s1, recent_ids=[], genome_getter=self._genome_getter)
        self.graph.add_strand(s2, recent_ids=["s1"], genome_getter=self._genome_getter)

        neighbors = self.graph.neighbors("s2")
        neighbor_ids = [n[0] for n in neighbors]
        assert "s1" in neighbor_ids

    def test_entity_shared_edges(self):
        reg_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        reg_file.close()
        os.unlink(reg_file.name)
        registry = EntityRegistry(path=reg_file.name)

        try:
            # Two strands sharing entity "alice"
            s1 = _make_strand("s1", entity_ids=["alice"])
            s2 = _make_strand("s2", entity_ids=["alice"])
            self._strands.update({"s1": s1, "s2": s2})

            # Register alice for both strands
            registry.resolve("Alice", EntityType.PERSON.value, "s1", 1000)
            registry.resolve("Alice", EntityType.PERSON.value, "s2", 2000)

            self.graph.add_strand(s1, recent_ids=[], entity_registry=registry,
                                  genome_getter=self._genome_getter)
            self.graph.add_strand(s2, recent_ids=[], entity_registry=registry,
                                  genome_getter=self._genome_getter)

            neighbors = self.graph.neighbors("s2")
            entity_neighbors = [n for n in neighbors if n[2] == "entity_shared"]
            assert len(entity_neighbors) > 0
        finally:
            if os.path.exists(reg_file.name):
                os.unlink(reg_file.name)

    def test_semantic_edges_same_domain_relation(self):
        s1 = _make_strand("s1", domain=Domain.SALES.value, relation=RelationType.WANTS.value)
        s2 = _make_strand("s2", domain=Domain.SALES.value, relation=RelationType.WANTS.value)
        self._strands.update({"s1": s1, "s2": s2})

        self.graph.add_strand(s1, recent_ids=[], genome_getter=self._genome_getter)
        self.graph.add_strand(s2, recent_ids=[], genome_getter=self._genome_getter)

        neighbors = self.graph.neighbors("s2")
        semantic_neighbors = [n for n in neighbors if n[2] == "semantic"]
        assert len(semantic_neighbors) > 0

    def test_no_semantic_edge_different_domain(self):
        s1 = _make_strand("s1", domain=Domain.SALES.value, relation=RelationType.WANTS.value)
        s2 = _make_strand("s2", domain=Domain.TECHNICAL.value, relation=RelationType.WANTS.value)
        self._strands.update({"s1": s1, "s2": s2})

        self.graph.add_strand(s1, recent_ids=[], genome_getter=self._genome_getter)
        self.graph.add_strand(s2, recent_ids=[], genome_getter=self._genome_getter)

        neighbors = self.graph.neighbors("s2")
        semantic_neighbors = [n for n in neighbors if n[2] == "semantic"]
        assert len(semantic_neighbors) == 0

    def test_hebbian_update_strengthens_edges(self):
        s1 = _make_strand("s1")
        s2 = _make_strand("s2")
        self._strands.update({"s1": s1, "s2": s2})
        self.graph.add_strand(s1, recent_ids=[], genome_getter=self._genome_getter)
        self.graph.add_strand(s2, recent_ids=["s1"], genome_getter=self._genome_getter)

        # Get initial weight
        initial = [w for nid, w, _ in self.graph.neighbors("s1") if nid == "s2"][0]

        # Hebbian update
        self.graph.hebbian_update(["s1", "s2"])
        after = [w for nid, w, _ in self.graph.neighbors("s1") if nid == "s2"][0]
        assert after > initial

    def test_hebbian_creates_causal_edge(self):
        # Use different domains so no semantic edge is auto-created
        s1 = _make_strand("s1", entity_ids=["alice"], domain=Domain.SALES.value, relation=RelationType.WANTS.value)
        s2 = _make_strand("s2", entity_ids=["bob"], domain=Domain.TECHNICAL.value, relation=RelationType.BLOCKS.value)
        self._strands.update({"s1": s1, "s2": s2})

        # Add without any shared edges (different entities, different domain+relation, no temporal)
        self.graph.add_strand(s1, recent_ids=[], genome_getter=self._genome_getter)
        self.graph.add_strand(s2, recent_ids=[], genome_getter=self._genome_getter)

        # Verify no edge initially
        neighbors_before = [n[0] for n in self.graph.neighbors("s1")]
        assert "s2" not in neighbors_before

        self.graph.hebbian_update(["s1", "s2"])
        neighbors_after = self.graph.neighbors("s1")
        causal = [n for n in neighbors_after if n[2] == "causal"]
        assert len(causal) > 0

    def test_decay(self):
        s1 = _make_strand("s1")
        s2 = _make_strand("s2")
        self._strands.update({"s1": s1, "s2": s2})
        self.graph.add_strand(s1, recent_ids=[], genome_getter=self._genome_getter)
        self.graph.add_strand(s2, recent_ids=["s1"], genome_getter=self._genome_getter)

        initial = [w for _, w, _ in self.graph.neighbors("s1") if True][0]
        self.graph.apply_decay()
        after = [w for _, w, _ in self.graph.neighbors("s1") if True][0]
        assert after < initial

    def test_ego_node(self):
        self.graph.ensure_ego_node("agent")
        assert self.graph.ego_node_count() == 1

        s1 = _make_strand("s1")
        self._strands["s1"] = s1
        self.graph.add_strand(s1, recent_ids=[], genome_getter=self._genome_getter)
        self.graph.link_to_ego("s1", "agent")

        ego_linked = self.graph.get_ego_linked_strands("agent")
        assert "s1" in ego_linked

    def test_recency_priming(self):
        self.graph.prime_recency(["s1"])
        assert self.graph.get_recency_bonus("s1") > 0
        self.graph.decay_recency()
        assert self.graph.get_recency_bonus("s1") < AssociationGraph.RECENCY_BONUS

    def test_edge_cap(self):
        """Adding many neighbors to one node should cap edges."""
        center = _make_strand("center")
        self._strands["center"] = center
        self.graph.add_strand(center, recent_ids=[], genome_getter=self._genome_getter)

        # Add many strands connected to center
        for i in range(self.graph.MAX_EDGES_PER_NODE + 20):
            sid = f"s{i}"
            s = _make_strand(sid)
            self._strands[sid] = s
            self.graph.add_strand(s, recent_ids=["center"], genome_getter=self._genome_getter)

        # Center's outgoing edges should be capped
        out_edges = list(self.graph.graph.edges("center"))
        assert len(out_edges) <= self.graph.MAX_EDGES_PER_NODE

    def test_decay_floor(self):
        """Edge weights should never decay below DECAY_FLOOR (R3 mitigation)."""
        s1 = _make_strand("s1")
        s2 = _make_strand("s2")
        self._strands.update({"s1": s1, "s2": s2})
        self.graph.add_strand(s1, recent_ids=[], genome_getter=self._genome_getter)
        self.graph.add_strand(s2, recent_ids=["s1"], genome_getter=self._genome_getter)

        # Apply decay many times — simulating 1000 queries
        for _ in range(1000):
            self.graph.apply_decay()

        # Weights should be at the floor, not zero
        for _, w, etype in self.graph.neighbors("s1"):
            if etype != "ego":
                assert w >= self.graph.DECAY_FLOOR

    def test_ego_edges_exempt_from_decay(self):
        """Ego edges should not decay (R3 mitigation)."""
        self.graph.ensure_ego_node("agent")
        s1 = _make_strand("s1")
        self._strands["s1"] = s1
        self.graph.add_strand(s1, recent_ids=[], genome_getter=self._genome_getter)
        self.graph.link_to_ego("s1", "agent")

        # Get initial ego edge weight
        ego_neighbors = [(nid, w, et) for nid, w, et in self.graph.neighbors("ego:agent")
                         if et == "ego"]
        assert len(ego_neighbors) > 0
        initial_weight = ego_neighbors[0][1]

        # Apply decay 100 times
        for _ in range(100):
            self.graph.apply_decay()

        ego_after = [(nid, w, et) for nid, w, et in self.graph.neighbors("ego:agent")
                     if et == "ego"]
        assert ego_after[0][1] == initial_weight  # unchanged

    def test_persistence(self):
        s1 = _make_strand("s1")
        self._strands["s1"] = s1
        self.graph.add_strand(s1, recent_ids=[], genome_getter=self._genome_getter)
        self.graph.ensure_ego_node("agent")

        loaded = AssociationGraph(path=self.tmpfile.name)
        assert loaded.node_count() == self.graph.node_count()
        assert loaded.edge_count() == self.graph.edge_count()
