"""Tests for the genome module — storage pipeline."""

import json
import os
import tempfile
import pytest
from cognitive_memory import (
    CodebookStrand, EntityType, RelationType, Modifier, TemporalMarker, Domain,
    make_codebook_strand, Genome, DNAEncoder,
)


def _make_strand(strand_id="s1", raw_hash="h1", timestamp=1000, relation=0):
    strand = make_codebook_strand(
        entity_slots=[(EntityType.PERSON.value, "alice")],
        relation=relation,
        modifier=Modifier.NEUTRAL.value,
        temporal=TemporalMarker.PRESENT.value,
        domain=Domain.SALES.value,
        sentiment=0,
        confidence=3,
        timestamp=timestamp,
        raw_hash=raw_hash,
    )
    strand.strand_id = strand_id
    return strand


class TestGenome:
    def setup_method(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmpfile.close()
        os.unlink(self.tmpfile.name)
        self.genome = Genome(path=self.tmpfile.name)

    def teardown_method(self):
        if os.path.exists(self.tmpfile.name):
            os.unlink(self.tmpfile.name)

    def test_add_and_get(self):
        strand = _make_strand()
        self.genome.add(strand)
        assert self.genome.get("s1") is not None
        assert self.genome.count() == 1

    def test_has_hash(self):
        strand = _make_strand(raw_hash="unique_hash")
        self.genome.add(strand)
        assert self.genome.has_hash("unique_hash")
        assert not self.genome.has_hash("other_hash")

    def test_remove(self):
        strand = _make_strand()
        self.genome.add(strand)
        self.genome.remove("s1")
        assert self.genome.get("s1") is None
        assert self.genome.count() == 0

    def test_supersede(self):
        s1 = _make_strand("s1", "h1")
        s2 = _make_strand("s2", "h2")
        self.genome.add(s1)
        self.genome.add(s2)
        self.genome.supersede("s1", "s2")
        assert self.genome.get("s1").superseded_by == "s2"
        assert len(self.genome.active_strands()) == 1

    def test_active_strands_excludes_superseded(self):
        s1 = _make_strand("s1", "h1")
        s2 = _make_strand("s2", "h2")
        self.genome.add(s1)
        self.genome.add(s2)
        self.genome.supersede("s1", "s2")
        active = self.genome.active_strands()
        assert len(active) == 1
        assert active[0].strand_id == "s2"

    def test_increment_activation(self):
        strand = _make_strand()
        self.genome.add(strand)
        assert self.genome.get("s1").activation_count == 0
        self.genome.increment_activation("s1")
        assert self.genome.get("s1").activation_count == 1

    def test_persistence(self):
        strand = _make_strand()
        self.genome.add(strand)
        # Load fresh
        loaded = Genome(path=self.tmpfile.name)
        assert loaded.count() == 1
        assert loaded.get("s1") is not None

    def test_batch_mode(self):
        self.genome.begin_batch()
        self.genome.add(_make_strand("s1", "h1"))
        self.genome.add(_make_strand("s2", "h2"))
        # File should not have been written yet
        if os.path.exists(self.tmpfile.name):
            with open(self.tmpfile.name) as f:
                data = json.load(f)
            # Might be empty from initial state
            assert len(data) == 0 or not os.path.exists(self.tmpfile.name)
        self.genome.end_batch()
        # Now verify it's persisted
        loaded = Genome(path=self.tmpfile.name)
        assert loaded.count() == 2

    def test_all_ids(self):
        self.genome.add(_make_strand("s1", "h1"))
        self.genome.add(_make_strand("s2", "h2"))
        assert set(self.genome.all_ids()) == {"s1", "s2"}


class TestDNAEncoderParsing:
    """Test the JSON parsing logic without API calls."""

    def test_parse_clean_json(self):
        text = '{"entities": [{"type": "PERSON", "name": "Alice"}], "relation": "WANTS", "modifier": "NEUTRAL", "temporal": "PRESENT", "domain": "SALES", "sentiment": 0, "confidence": 4, "trace": "Alice wants something"}'
        result = DNAEncoder._parse_llm_json(text)
        assert result["entities"][0]["name"] == "Alice"
        assert result["relation"] == "WANTS"

    def test_parse_markdown_fenced(self):
        text = '```json\n{"entities": [], "relation": "WANTS"}\n```'
        result = DNAEncoder._parse_llm_json(text)
        assert result["relation"] == "WANTS"

    def test_parse_extra_text(self):
        text = 'Here is the result: {"entities": [], "relation": "BLOCKS"} hope that helps!'
        result = DNAEncoder._parse_llm_json(text)
        assert result["relation"] == "BLOCKS"

    def test_parse_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            DNAEncoder._parse_llm_json("not json at all")
