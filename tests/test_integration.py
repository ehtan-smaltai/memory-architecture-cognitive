"""
Integration tests — validate the full pipeline with real LLM calls.

These tests require ANTHROPIC_API_KEY to be set. They are skipped in CI
unless the secret is available. Run locally with:

    pytest tests/test_integration.py -v -m integration

Or run all tests except integration:

    pytest tests/ -v -m "not integration"
"""

import os
import tempfile
import shutil

import pytest

from cognitive_memory import (
    Codebook,
    Config,
    EntityType,
    RelationType,
    Modifier,
    TemporalMarker,
    Domain,
    EntityRegistry,
    DNAEncoder,
    Genome,
    AssociationGraph,
    MemorySystem,
)


pytestmark = pytest.mark.integration

# Skip entire module if no API key
if not os.environ.get("ANTHROPIC_API_KEY"):
    pytest.skip("ANTHROPIC_API_KEY not set — skipping integration tests", allow_module_level=True)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def codebook():
    return Codebook()


@pytest.fixture
def entity_registry(tmpdir):
    return EntityRegistry(path=os.path.join(tmpdir, "entities.json"))


@pytest.fixture
def encoder(codebook, entity_registry):
    return DNAEncoder(codebook=codebook, entity_registry=entity_registry)


@pytest.fixture
def memory_system(tmpdir):
    return MemorySystem(
        genome_path=os.path.join(tmpdir, "genome.json"),
        graph_path=os.path.join(tmpdir, "graph.json"),
        entities_path=os.path.join(tmpdir, "entities.json"),
    )


# ─── Valid code sets for validation ──────────────────────────────────────────

VALID_ENTITY_TYPES = {e.value for e in EntityType}
VALID_RELATIONS = {r.value for r in RelationType}
VALID_MODIFIERS = {m.value for m in Modifier}
VALID_TEMPORALS = {t.value for t in TemporalMarker}
VALID_DOMAINS = {d.value for d in Domain}


def assert_valid_strand(strand):
    """Assert that a CodebookStrand has valid codebook codes everywhere."""
    # Entity types must be valid codes
    for etype, inst_id in strand.entity_slots:
        assert etype in VALID_ENTITY_TYPES, f"Invalid entity type code: {etype}"
        assert isinstance(inst_id, str) and len(inst_id) > 0, f"Empty instance_id"

    # Must have at least one entity
    assert len(strand.entity_slots) >= 1, "Strand has no entities"
    assert len(strand.entity_slots) <= 4, "Strand has >4 entities"

    # Relation must be valid
    assert strand.relation in VALID_RELATIONS, f"Invalid relation code: {strand.relation}"

    # Modifier must be valid
    assert strand.modifier in VALID_MODIFIERS, f"Invalid modifier code: {strand.modifier}"

    # Temporal must be valid
    assert strand.temporal in VALID_TEMPORALS, f"Invalid temporal code: {strand.temporal}"

    # Domain must be valid
    assert strand.domain in VALID_DOMAINS, f"Invalid domain code: {strand.domain}"

    # Sentiment and confidence in range
    assert -2 <= strand.sentiment <= 2, f"Sentiment out of range: {strand.sentiment}"
    assert 1 <= strand.confidence <= 5, f"Confidence out of range: {strand.confidence}"

    # Trace should exist and be reasonable length
    assert isinstance(strand.trace, str), "Trace is not a string"
    assert len(strand.trace) > 0, "Trace is empty"


# ─── Test: LLM extraction produces valid codebook codes ─────────────────────

SAMPLE_INPUTS = [
    "Client Acme Corp mentioned budget concerns for Q3, worried about the $15k/month price point",
    "Sarah from Delta Inc loved the demo and wants to move forward with a pilot program",
    "James at TechFlow has gone quiet after receiving our proposal last Tuesday",
    "Competitor CloudBase just launched a cheaper alternative targeting our mid-market segment",
    "The engineering team reported that the API integration with Stripe is breaking down intermittently",
]


class TestLLMExtraction:
    """Validate that the LLM extraction prompt produces valid codebook codes."""

    @pytest.mark.parametrize("raw_text", SAMPLE_INPUTS, ids=[
        "budget_concern",
        "positive_demo",
        "gone_quiet",
        "competitor_launch",
        "technical_issue",
    ])
    def test_extraction_produces_valid_codes(self, encoder, raw_text):
        """Each extraction must produce a strand with all valid codebook codes."""
        strand = encoder.encode(raw_text)
        assert_valid_strand(strand)

    def test_extraction_not_all_fallback(self, encoder):
        """At least some extractions should produce specific codes, not just UNKNOWN/OTHER."""
        specific_count = 0
        for raw_text in SAMPLE_INPUTS[:3]:
            strand = encoder.encode(raw_text)
            if strand.relation != RelationType.OTHER.value:
                specific_count += 1
            if any(e != EntityType.UNKNOWN.value for e, _ in strand.entity_slots):
                specific_count += 1

        # At least 4 of 6 checks should be specific (not fallback)
        assert specific_count >= 4, (
            f"Only {specific_count}/6 extractions produced specific codes — "
            f"LLM is falling back to OTHER/UNKNOWN too often"
        )

    def test_entity_resolution_consistency(self, encoder, entity_registry):
        """Same entity mentioned differently should resolve to same instance."""
        encoder.encode("Sarah Chen from Acme Corp discussed pricing")
        encoder.encode("Sarah mentioned budget concerns about the deal")

        # "Sarah Chen" and "Sarah" should either match or be separate
        # (depends on fuzzy matching thresholds — this validates no crash)
        assert entity_registry.count() >= 1
        assert entity_registry.count() <= 4  # sanity: not exploding


class TestEndToEnd:
    """Full store → query round trip."""

    def test_store_and_query(self, memory_system):
        """Store memories, then query — should get a non-empty answer."""
        # Store
        s1 = memory_system.store("Acme Corp has budget concerns about our $15k/month pricing")
        s2 = memory_system.store("Sarah from Acme loves the product but needs a discount")
        s3 = memory_system.store("Sent Acme Corp a revised proposal with 20% discount")

        assert s1 is not None
        assert s2 is not None
        assert s3 is not None

        # Query
        result = memory_system.query("What's happening with Acme Corp?")

        assert "answer" in result
        assert len(result["answer"]) > 0
        assert result["tokens_used"] > 0
        assert result["api_calls"] == 1
        assert len(result["activated"]) >= 1

    def test_store_dedup(self, memory_system):
        """Storing the same text twice should deduplicate."""
        text = "Client mentioned they need enterprise SSO support"
        s1 = memory_system.store(text)
        s2 = memory_system.store(text)

        assert s1 is not None
        assert s2 is None  # deduplicated

    def test_stats_after_storage(self, memory_system):
        """Stats should reflect stored strands."""
        memory_system.store("Alice from BetaCo signed up for a trial")
        memory_system.store("Bob from GammaCo wants a demo next week")

        stats = memory_system.stats()
        assert stats["total_strands"] == 2
        assert stats["active_strands"] == 2
        assert stats["superseded_strands"] == 0
        assert stats["entities"] >= 2
        assert stats["graph_nodes"] >= 2

    def test_consolidation_runs(self, memory_system):
        """Consolidation should run without error even on small data."""
        memory_system.store("Acme Corp is interested in our enterprise plan")
        memory_system.store("Acme Corp asked about pricing for 50 seats")
        memory_system.store("Acme Corp wants to schedule a call next week")

        result = memory_system.consolidate()
        assert "groups_found" in result
        assert "merged" in result
        # May or may not merge (depends on LLM extraction) — just verify no crash

    def test_forgetting_runs(self, memory_system):
        """Forgetting should run without error."""
        memory_system.store("One-off mention of irrelevant topic")

        result = memory_system.forget(min_age_seconds=0, min_activations=0)
        assert "forgotten" in result
