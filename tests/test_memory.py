"""Tests for the MemorySystem — consolidation, forgetting, superseding (no API calls)."""

import os
import tempfile
import pytest
from codebook import (
    CodebookStrand, EntityType, RelationType, Modifier,
    TemporalMarker, Domain, make_codebook_strand,
)
from genome import Genome
from graph import AssociationGraph
from entities import EntityRegistry
from memory import MemorySystem


def _make_strand(strand_id, entity_ids=None, relation=0, modifier=0,
                 domain=0, sentiment=0, confidence=3, timestamp=1000, raw_hash=None):
    if entity_ids is None:
        entity_ids = ["alice"]
    strand = make_codebook_strand(
        entity_slots=[(EntityType.PERSON.value, eid) for eid in entity_ids],
        relation=relation,
        modifier=modifier,
        temporal=TemporalMarker.PRESENT.value,
        domain=domain,
        sentiment=sentiment,
        confidence=confidence,
        timestamp=timestamp,
        raw_hash=raw_hash or strand_id,
    )
    strand.strand_id = strand_id
    return strand


class TestMemoryConsolidation:
    """Test consolidation without API calls by injecting strands directly."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.genome = Genome(path=os.path.join(self.tmpdir, "genome.json"))
        self.graph = AssociationGraph(path=os.path.join(self.tmpdir, "graph.json"))
        self.registry = EntityRegistry(path=os.path.join(self.tmpdir, "entities.json"))

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _build_system_with_strands(self, strands):
        """Helper to create a MemorySystem-like setup with pre-injected strands."""
        from codebook import Codebook
        codebook = Codebook()

        for s in strands:
            self.genome.add(s)
            self.graph.add_strand(s, recent_ids=[], genome_getter=self.genome.get)

        # Create a minimal MemorySystem-like object for consolidation/forgetting
        # We test the methods directly on the real MemorySystem components
        return codebook

    def test_consolidation_merges_related_strands(self):
        """3+ strands with same entity + same relation cluster → consolidate."""
        from memory import MemorySystem

        # Create 3 strands for "alice" with PRICE_CONCERN (cluster 0)
        strands = [
            _make_strand(f"s{i}", ["alice"],
                         relation=RelationType.PRICE_CONCERN.value,
                         domain=Domain.SALES.value,
                         timestamp=1000 + i * 100,
                         raw_hash=f"h{i}")
            for i in range(3)
        ]
        strands[0].trace = "Budget issues Q1"
        strands[1].trace = "Pricing too high"
        strands[2].trace = "Discount requested"

        codebook = self._build_system_with_strands(strands)

        # Build a minimal system object to test consolidate
        system = object.__new__(MemorySystem)
        system.codebook = codebook
        system.genome = self.genome
        system.graph = self.graph
        system.entity_registry = self.registry

        result = system.consolidate()
        assert result["consolidated"] >= 2  # at least 2 old strands superseded

        # The most recent strand should be the keeper
        active = self.genome.active_strands()
        assert len(active) == 1
        assert "Budget issues Q1" in active[0].trace or "Pricing too high" in active[0].trace

    def test_consolidation_skips_small_groups(self):
        """Groups with < 3 strands should not be consolidated."""
        from memory import MemorySystem

        strands = [
            _make_strand("s1", ["alice"], relation=RelationType.WANTS.value, raw_hash="h1"),
            _make_strand("s2", ["alice"], relation=RelationType.WANTS.value, raw_hash="h2"),
        ]
        codebook = self._build_system_with_strands(strands)

        system = object.__new__(MemorySystem)
        system.codebook = codebook
        system.genome = self.genome
        system.graph = self.graph
        system.entity_registry = self.registry

        result = system.consolidate()
        assert result["consolidated"] == 0

    def test_forgetting_removes_unused(self):
        """Strands with 0 activations older than threshold get forgotten."""
        from memory import MemorySystem

        strands = [
            _make_strand("s1", ["alice"], timestamp=1000, raw_hash="h1"),
            _make_strand("s2", ["alice"], timestamp=2000, raw_hash="h2"),
            _make_strand("s3", ["bob"], timestamp=3000, raw_hash="h3"),
        ]
        # s1 has been activated, s2 and s3 haven't
        strands[0].activation_count = 5

        codebook = self._build_system_with_strands(strands)

        system = object.__new__(MemorySystem)
        system.codebook = codebook
        system.genome = self.genome
        system.graph = self.graph
        system.entity_registry = self.registry

        result = system.forget(min_age_seconds=0, min_activations=0)
        # s2 can be forgotten (alice has other refs), s3 cannot (only ref for bob)
        assert result["forgotten"] >= 1
        assert self.genome.get("s1") is not None  # activated — protected

    def test_supersede_logic(self):
        """Same entity + same relation + different sentiment → supersede."""
        from memory import MemorySystem

        s1 = _make_strand("s1", ["alice"], relation=RelationType.PRICE_CONCERN.value,
                          sentiment=-2, raw_hash="h1")
        s2 = _make_strand("s2", ["alice"], relation=RelationType.PRICE_CONCERN.value,
                          sentiment=1, raw_hash="h2")

        self.genome.add(s1)
        self.graph.add_strand(s1, recent_ids=[], genome_getter=self.genome.get)

        codebook = self._build_system_with_strands([])

        system = object.__new__(MemorySystem)
        system.codebook = codebook
        system.genome = self.genome
        system.graph = self.graph
        system.entity_registry = self.registry

        system._check_supersede(s2)
        assert self.genome.get("s1").superseded_by == "s2"

    def test_supersede_skips_factual_relations(self):
        """Factual relations like SENT should NOT be superseded (R5 mitigation)."""
        from memory import MemorySystem

        # SENT is a historical event, not a belief state
        s1 = _make_strand("s1", ["alice"], relation=RelationType.SENT.value,
                          sentiment=0, raw_hash="h1")
        s2 = _make_strand("s2", ["alice"], relation=RelationType.SENT.value,
                          sentiment=1, raw_hash="h2")

        self.genome.add(s1)
        self.graph.add_strand(s1, recent_ids=[], genome_getter=self.genome.get)

        codebook = self._build_system_with_strands([])

        system = object.__new__(MemorySystem)
        system.codebook = codebook
        system.genome = self.genome
        system.graph = self.graph
        system.entity_registry = self.registry

        system._check_supersede(s2)
        assert self.genome.get("s1").superseded_by is None  # NOT superseded

    def test_supersede_works_for_belief_states(self):
        """Belief-state relations like HESITANT should be superseded (R5 mitigation)."""
        from memory import MemorySystem

        s1 = _make_strand("s1", ["alice"], relation=RelationType.HESITANT.value,
                          sentiment=-1, raw_hash="h1")
        s2 = _make_strand("s2", ["alice"], relation=RelationType.HESITANT.value,
                          sentiment=1, raw_hash="h2")

        self.genome.add(s1)
        self.graph.add_strand(s1, recent_ids=[], genome_getter=self.genome.get)

        codebook = self._build_system_with_strands([])

        system = object.__new__(MemorySystem)
        system.codebook = codebook
        system.genome = self.genome
        system.graph = self.graph
        system.entity_registry = self.registry

        system._check_supersede(s2)
        assert self.genome.get("s1").superseded_by == "s2"  # IS superseded
