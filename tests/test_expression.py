"""Tests for the expression engine — spreading activation, similarity, token estimation."""

import pytest
from cognitive_memory import (
    Codebook, CodebookStrand, EntityType, RelationType, Modifier,
    TemporalMarker, Domain, make_codebook_strand,
    codebook_similarity, estimate_strand_tokens,
)


def _make_strand(entity_ids=None, relation=0, modifier=0, domain=0, temporal=0):
    if entity_ids is None:
        entity_ids = ["alice"]
    return make_codebook_strand(
        entity_slots=[(EntityType.PERSON.value, eid) for eid in entity_ids],
        relation=relation,
        modifier=modifier,
        temporal=temporal,
        domain=domain,
        sentiment=0,
        confidence=3,
        timestamp=1000,
        raw_hash="test",
    )


class TestCodebookSimilarity:
    def setup_method(self):
        self.cb = Codebook()

    def test_identical_strands(self):
        s = _make_strand(["alice"], RelationType.WANTS.value, Modifier.URGENT.value, Domain.SALES.value)
        sim = codebook_similarity(s, s, self.cb)
        assert sim == pytest.approx(1.0)

    def test_same_entity_different_relation(self):
        s1 = _make_strand(["alice"], RelationType.WANTS.value, Modifier.NEUTRAL.value, Domain.SALES.value)
        s2 = _make_strand(["alice"], RelationType.BLOCKS.value, Modifier.NEUTRAL.value, Domain.SALES.value)
        sim = codebook_similarity(s1, s2, self.cb)
        # Same entity (0.4) + same domain (0.25) + different relation (0.0) + same modifier (0.15*1.0)
        assert sim > 0.5

    def test_no_overlap(self):
        s1 = _make_strand(["alice"], RelationType.WANTS.value, Modifier.URGENT.value, Domain.SALES.value)
        s2 = _make_strand(["bob"], RelationType.CANCELLED.value, Modifier.POSITIVE.value, Domain.TECHNICAL.value)
        sim = codebook_similarity(s1, s2, self.cb)
        assert sim == 0.0

    def test_cluster_relation_similarity(self):
        """PRICE_CONCERN and DISCOUNT_REQ are in the same cluster."""
        s1 = _make_strand(["alice"], RelationType.PRICE_CONCERN.value, domain=Domain.SALES.value)
        s2 = _make_strand(["alice"], RelationType.DISCOUNT_REQ.value, domain=Domain.SALES.value)
        sim = codebook_similarity(s1, s2, self.cb)
        # entity(0.4*1.0) + domain(0.25*1.0) + relation(0.20*0.7) + modifier(0.15*1.0)
        assert sim > 0.8

    def test_empty_entities(self):
        s1 = make_codebook_strand(
            entity_slots=[], relation=0, modifier=0, temporal=0, domain=0,
            sentiment=0, confidence=3, timestamp=0, raw_hash="a",
        )
        s2 = make_codebook_strand(
            entity_slots=[], relation=0, modifier=0, temporal=0, domain=0,
            sentiment=0, confidence=3, timestamp=0, raw_hash="b",
        )
        sim = codebook_similarity(s1, s2, self.cb)
        # No entities → entity_score=0, but domain, relation, modifier all match
        assert sim > 0


class TestTokenEstimation:
    def test_strand_without_trace(self):
        strand = _make_strand()
        strand.trace = ""
        tokens = estimate_strand_tokens(strand)
        assert tokens > 0
        assert tokens == strand.sequence_length() + 10

    def test_strand_with_trace(self):
        strand = _make_strand()
        strand.trace = "Alice wants to buy the product for ten thousand dollars"
        tokens_with = estimate_strand_tokens(strand)
        strand.trace = ""
        tokens_without = estimate_strand_tokens(strand)
        assert tokens_with > tokens_without

    def test_more_entities_more_tokens(self):
        s1 = _make_strand(["alice"])
        s2 = _make_strand(["alice", "bob", "charlie"])
        assert estimate_strand_tokens(s2) > estimate_strand_tokens(s1)
