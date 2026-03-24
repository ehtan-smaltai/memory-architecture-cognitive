"""Tests for the codebook module — the DNA alphabet."""

import pytest
from codebook import (
    Codebook,
    CodebookStrand,
    EntityType,
    RelationType,
    Modifier,
    TemporalMarker,
    Domain,
    make_codebook_strand,
)


class TestCodebook:
    def setup_method(self):
        self.cb = Codebook()

    def test_encode_entity_type_known(self):
        assert self.cb.encode_entity_type("PERSON") == EntityType.PERSON.value
        assert self.cb.encode_entity_type("org") == EntityType.ORG.value
        assert self.cb.encode_entity_type(" Product ") == EntityType.PRODUCT.value

    def test_encode_entity_type_unknown(self):
        assert self.cb.encode_entity_type("NONSENSE") == EntityType.UNKNOWN.value

    def test_encode_relation_known(self):
        assert self.cb.encode_relation("WANTS") == RelationType.WANTS.value
        assert self.cb.encode_relation("price_concern") == RelationType.PRICE_CONCERN.value

    def test_encode_relation_fallback(self):
        assert self.cb.encode_relation("NONEXISTENT") == RelationType.OTHER.value

    def test_encode_modifier(self):
        assert self.cb.encode_modifier("URGENT") == Modifier.URGENT.value
        assert self.cb.encode_modifier("junk") == Modifier.OTHER.value

    def test_encode_temporal(self):
        assert self.cb.encode_temporal("PAST") == TemporalMarker.PAST.value
        assert self.cb.encode_temporal("junk") == TemporalMarker.PRESENT.value

    def test_encode_domain(self):
        assert self.cb.encode_domain("SALES") == Domain.SALES.value
        assert self.cb.encode_domain("junk") == Domain.GENERAL.value

    def test_decode_roundtrip(self):
        for e in EntityType:
            assert self.cb.decode_entity_type(e.value) == e.name or e == EntityType.UNKNOWN

    def test_relation_similarity_same(self):
        assert self.cb.relation_similarity(RelationType.WANTS.value, RelationType.WANTS.value) == 1.0

    def test_relation_similarity_cluster(self):
        # PRICE_CONCERN and DISCOUNT_REQ are in the same cluster
        sim = self.cb.relation_similarity(RelationType.PRICE_CONCERN.value, RelationType.DISCOUNT_REQ.value)
        assert sim == 0.7

    def test_relation_similarity_different(self):
        sim = self.cb.relation_similarity(RelationType.WANTS.value, RelationType.CANCELLED.value)
        assert sim == 0.0

    def test_modifier_similarity_same(self):
        assert self.cb.modifier_similarity(Modifier.URGENT.value, Modifier.URGENT.value) == 1.0

    def test_modifier_similarity_aligned(self):
        # URGENT and NEGATIVE are both in the negative group
        sim = self.cb.modifier_similarity(Modifier.URGENT.value, Modifier.NEGATIVE.value)
        assert sim == 0.6

    def test_modifier_similarity_different(self):
        sim = self.cb.modifier_similarity(Modifier.POSITIVE.value, Modifier.NEGATIVE.value)
        assert sim == 0.0

    def test_total_codes(self):
        total = self.cb.total_codes()
        assert total > 0
        assert isinstance(total, int)

    def test_list_methods_nonempty(self):
        assert len(self.cb.entity_type_list()) > 0
        assert len(self.cb.relation_list()) > 0
        assert len(self.cb.modifier_list()) > 0
        assert len(self.cb.temporal_list()) > 0
        assert len(self.cb.domain_list()) > 0


class TestCodebookStrand:
    def test_to_sequence(self):
        strand = make_codebook_strand(
            entity_slots=[(EntityType.PERSON.value, "alice"), (EntityType.ORG.value, "acme")],
            relation=RelationType.WANTS.value,
            modifier=Modifier.URGENT.value,
            temporal=TemporalMarker.PRESENT.value,
            domain=Domain.SALES.value,
            sentiment=1,
            confidence=4,
            timestamp=1000,
            raw_hash="abc123",
        )
        seq = strand.to_sequence()
        # 2 entities + 6 fields = 8
        assert len(seq) == 8
        assert seq[0] == EntityType.PERSON.value  # first entity type
        assert seq[1] == EntityType.ORG.value      # second entity type
        assert seq[2] == 100 + RelationType.WANTS.value
        assert seq[3] == 200 + Modifier.URGENT.value

    def test_sequence_length(self):
        strand = make_codebook_strand(
            entity_slots=[(0, "x")],
            relation=0, modifier=0, temporal=0, domain=0,
            sentiment=0, confidence=3, timestamp=0, raw_hash="h",
        )
        assert strand.sequence_length() == 7  # 1 entity + 6

    def test_to_dict_from_dict_roundtrip(self):
        strand = make_codebook_strand(
            entity_slots=[(EntityType.PERSON.value, "alice")],
            relation=RelationType.WANTS.value,
            modifier=Modifier.POSITIVE.value,
            temporal=TemporalMarker.FUTURE.value,
            domain=Domain.TECHNICAL.value,
            sentiment=-1,
            confidence=3,
            timestamp=12345,
            raw_hash="deadbeef",
        )
        strand.trace = "Alice wants a technical thing"
        strand.activation_count = 5

        d = strand.to_dict()
        restored = CodebookStrand.from_dict(d)

        assert restored.strand_id == strand.strand_id
        assert restored.entity_slots == strand.entity_slots
        assert restored.relation == strand.relation
        assert restored.modifier == strand.modifier
        assert restored.trace == strand.trace
        assert restored.activation_count == 5

    def test_sentiment_clamped(self):
        strand = make_codebook_strand(
            entity_slots=[], relation=0, modifier=0, temporal=0, domain=0,
            sentiment=10, confidence=99, timestamp=0, raw_hash="h",
        )
        assert strand.sentiment == 2
        assert strand.confidence == 5

    def test_get_entity_instance_ids(self):
        strand = make_codebook_strand(
            entity_slots=[(0, "alice"), (1, "acme")],
            relation=0, modifier=0, temporal=0, domain=0,
            sentiment=0, confidence=3, timestamp=0, raw_hash="h",
        )
        assert strand.get_entity_instance_ids() == ["alice", "acme"]

    def test_get_entity_types(self):
        strand = make_codebook_strand(
            entity_slots=[(EntityType.PERSON.value, "a"), (EntityType.ORG.value, "b")],
            relation=0, modifier=0, temporal=0, domain=0,
            sentiment=0, confidence=3, timestamp=0, raw_hash="h",
        )
        assert strand.get_entity_types() == [EntityType.PERSON.value, EntityType.ORG.value]
