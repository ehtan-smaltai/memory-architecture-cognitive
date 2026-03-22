"""
Codebook — The Finite Alphabet of Cognitive Memory

Analogous to DNA's 4 nucleotides (A, T, G, C) encoding all biological
information, this codebook defines a fixed, finite vocabulary of semantic
primitives. Every memory strand is encoded as a sequence of codebook indices.

Hierarchy (mirrors molecular biology):
  Nucleotide → individual codebook codes (EntityType, RelationType, etc.)
  Codon      → a CodebookStrand (one complete encoded memory unit)
  Gene       → a cluster of related strands (same entity/topic)
  Chromosome → all strands in the genome

~200 total codes across 5 enums. Fixed-width encoding enables real compression.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import IntEnum


# ─── Nucleotides: The 5 code dimensions ─────────────────────────────────────

class EntityType(IntEnum):
    """What kind of entity is being referenced."""
    PERSON = 0
    ORG = 1
    PRODUCT = 2
    METRIC = 3
    LOCATION = 4
    EVENT = 5
    DOCUMENT = 6
    TOOL = 7
    TEAM = 8
    ROLE = 9
    PROJECT = 10
    FEATURE = 11
    COMPETITOR = 12
    CURRENCY = 13
    DATE_REF = 14
    CHANNEL = 15
    UNKNOWN = 99


class RelationType(IntEnum):
    """The action or relationship between entities."""
    WANTS = 0
    BLOCKS = 1
    ENABLES = 2
    CONCERNS = 3
    REQUESTS = 4
    CONFIRMS = 5
    EVALUATES = 6
    REPORTS_TO = 7
    SIGNED_UP = 8
    EXPRESSED = 9
    ASKED_ABOUT = 10
    MENTIONED = 11
    LAUNCHED = 12
    SENT = 13
    PREFERS = 14
    DECLINED = 15
    ESCALATED = 16
    RENEWED = 17
    CANCELLED = 18
    COMPARED = 19
    INTEGRATED = 20
    DELAYED = 21
    APPROVED = 22
    REJECTED = 23
    SCHEDULED = 24
    FEEDBACK = 25
    PRICE_CONCERN = 26
    WENT_QUIET = 27
    RE_ENGAGED = 28
    TRIAL_STARTED = 29
    TRIAL_POSITIVE = 30
    TRIAL_NEGATIVE = 31
    DISCOUNT_REQ = 32
    REFERENCE_REQ = 33
    PROPOSAL_SENT = 34
    BUDGET_CYCLE = 35
    EXPANDING = 36
    BREAKING_DOWN = 37
    HESITANT = 38
    COMPETING = 39
    OTHER = 99


class Modifier(IntEnum):
    """Contextual modifiers — sentiment, urgency, significance."""
    URGENT = 0
    POSITIVE = 1
    NEGATIVE = 2
    NEUTRAL = 3
    UNCERTAIN = 4
    DEADLINE = 5
    HIGH_VALUE = 6
    LOW_VALUE = 7
    COMPETITIVE = 8
    RECURRING = 9
    ESCALATION = 10
    RESOLUTION = 11
    OTHER = 15


class TemporalMarker(IntEnum):
    """When does this memory relate to."""
    PAST = 0
    PRESENT = 1
    FUTURE = 2
    RECURRING = 3
    DEADLINE = 4


class Domain(IntEnum):
    """Coarse domain classification."""
    SALES = 0
    TECHNICAL = 1
    OPS = 2
    HR = 3
    FINANCE = 4
    LEGAL = 5
    MARKETING = 6
    SUPPORT = 7
    GENERAL = 8
    PRODUCT = 9


# ─── Codon: The CodebookStrand ──────────────────────────────────────────────

@dataclass
class CodebookStrand:
    """
    A single encoded memory unit — the 'codon' of the cognitive genome.

    Fixed-width: entity_slots + relation + modifier + temporal + domain +
    quantized sentiment + quantized confidence. No variable-length text.
    """
    strand_id: str
    entity_slots: list[tuple[int, str]]  # [(EntityType.value, instance_id), ...]
    relation: int                         # RelationType.value
    modifier: int                         # Modifier.value
    temporal: int                         # TemporalMarker.value
    domain: int                           # Domain.value
    sentiment: int                        # quantized: -2, -1, 0, 1, 2
    confidence: int                       # quantized: 1-5
    timestamp: int
    raw_hash: str
    trace: str = ""                       # neocortical trace — 10-15 word micro-summary
    activation_count: int = 0             # how many times this strand has been activated
    superseded_by: str | None = None      # strand_id that supersedes this one

    def to_sequence(self) -> list[int]:
        """
        Serialize to a flat list of codebook indices.
        This IS the DNA sequence — a fixed-width integer array.

        Encoding scheme (offset ranges prevent collision):
          0-99:   entity type codes
          100-199: relation codes
          200-215: modifier codes
          300-304: temporal codes
          400-409: domain codes
          500-504: sentiment (mapped -2..2 → 500..504)
          600-605: confidence (mapped 1..5 → 601..605)
        """
        seq = []
        for etype, _instance_id in self.entity_slots:
            seq.append(int(etype))
        seq.append(100 + int(self.relation))
        seq.append(200 + int(self.modifier))
        seq.append(300 + int(self.temporal))
        seq.append(400 + int(self.domain))
        seq.append(500 + self.sentiment + 2)
        seq.append(600 + self.confidence)
        return seq

    def sequence_length(self) -> int:
        """Fixed-width length of this strand's DNA sequence."""
        return len(self.entity_slots) + 6  # entities + relation + modifier + temporal + domain + sentiment + confidence

    def to_dict(self) -> dict:
        """Serialize for JSON persistence."""
        return {
            "strand_id": self.strand_id,
            "entity_slots": [[etype, inst_id] for etype, inst_id in self.entity_slots],
            "relation": self.relation,
            "modifier": self.modifier,
            "temporal": self.temporal,
            "domain": self.domain,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "raw_hash": self.raw_hash,
            "trace": self.trace,
            "activation_count": self.activation_count,
            "superseded_by": self.superseded_by,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CodebookStrand:
        """Deserialize from JSON."""
        return cls(
            strand_id=d["strand_id"],
            entity_slots=[(e[0], e[1]) for e in d["entity_slots"]],
            relation=d["relation"],
            modifier=d["modifier"],
            temporal=d["temporal"],
            domain=d["domain"],
            sentiment=d["sentiment"],
            confidence=d["confidence"],
            timestamp=d["timestamp"],
            raw_hash=d["raw_hash"],
            trace=d.get("trace", ""),
            activation_count=d.get("activation_count", 0),
            superseded_by=d.get("superseded_by"),
        )

    def get_entity_instance_ids(self) -> list[str]:
        """Return all entity instance IDs referenced by this strand."""
        return [inst_id for _, inst_id in self.entity_slots]

    def get_entity_types(self) -> list[int]:
        """Return all entity type codes referenced by this strand."""
        return [etype for etype, _ in self.entity_slots]


# ─── Codebook: The lookup table ─────────────────────────────────────────────

class Codebook:
    """
    The finite alphabet. Maps between human-readable names and integer codes.
    Stored once, shared across all strands — like a codon table.
    """

    def __init__(self):
        self._entity_types = {e.name: e.value for e in EntityType}
        self._relations = {r.name: r.value for r in RelationType}
        self._modifiers = {m.name: m.value for m in Modifier}
        self._temporals = {t.name: t.value for t in TemporalMarker}
        self._domains = {d.name: d.value for d in Domain}

        # Reverse maps for decode
        self._entity_type_names = {e.value: e.name for e in EntityType}
        self._relation_names = {r.value: r.name for r in RelationType}
        self._modifier_names = {m.value: m.name for m in Modifier}
        self._temporal_names = {t.value: t.name for t in TemporalMarker}
        self._domain_names = {d.value: d.name for d in Domain}

    def encode_entity_type(self, raw: str) -> int:
        """Map a string entity type to its codebook integer."""
        normalized = raw.upper().strip().replace(" ", "_")
        return self._entity_types.get(normalized, EntityType.UNKNOWN.value)

    def encode_relation(self, raw: str) -> int:
        normalized = raw.upper().strip().replace(" ", "_")
        return self._relations.get(normalized, RelationType.OTHER.value)

    def encode_modifier(self, raw: str) -> int:
        normalized = raw.upper().strip().replace(" ", "_")
        return self._modifiers.get(normalized, Modifier.OTHER.value)

    def encode_temporal(self, raw: str) -> int:
        normalized = raw.upper().strip().replace(" ", "_")
        return self._temporals.get(normalized, TemporalMarker.PRESENT.value)

    def encode_domain(self, raw: str) -> int:
        normalized = raw.upper().strip().replace(" ", "_")
        return self._domains.get(normalized, Domain.GENERAL.value)

    def decode_entity_type(self, code: int) -> str:
        return self._entity_type_names.get(code, "UNKNOWN")

    def decode_relation(self, code: int) -> str:
        return self._relation_names.get(code, "OTHER")

    def decode_modifier(self, code: int) -> str:
        return self._modifier_names.get(code, "OTHER")

    def decode_temporal(self, code: int) -> str:
        return self._temporal_names.get(code, "PRESENT")

    def decode_domain(self, code: int) -> str:
        return self._domain_names.get(code, "GENERAL")

    def entity_type_list(self) -> list[str]:
        """All entity type names — used in extraction prompts."""
        return [e.name for e in EntityType if e != EntityType.UNKNOWN]

    def relation_list(self) -> list[str]:
        return [r.name for r in RelationType if r != RelationType.OTHER]

    def modifier_list(self) -> list[str]:
        return [m.name for m in Modifier if m != Modifier.OTHER]

    def temporal_list(self) -> list[str]:
        return [t.name for t in TemporalMarker]

    def domain_list(self) -> list[str]:
        return [d.name for d in Domain]

    def total_codes(self) -> int:
        """Total number of codes in the alphabet."""
        return (
            len(EntityType)
            + len(RelationType)
            + len(Modifier)
            + len(TemporalMarker)
            + len(Domain)
            + 5  # sentiment levels
            + 5  # confidence levels
        )

    # ── Code Similarity Matrix ───────────────────────────────────────────
    # The brain doesn't match by exact codes — it matches by MEANING.
    # "budget concerns" and "pricing issues" are semantically related
    # even though they map to different codes.

    # Relation clusters — codes that are semantically close
    _RELATION_CLUSTERS: dict[int, set[int]] = {
        # Pricing/money cluster
        0: {RelationType.PRICE_CONCERN, RelationType.DISCOUNT_REQ,
            RelationType.BUDGET_CYCLE, RelationType.HESITANT},
        # Engagement cluster
        1: {RelationType.WANTS, RelationType.REQUESTS, RelationType.ASKED_ABOUT,
            RelationType.RE_ENGAGED},
        # Positive progress cluster
        2: {RelationType.TRIAL_STARTED, RelationType.TRIAL_POSITIVE,
            RelationType.FEEDBACK, RelationType.SIGNED_UP},
        # Negative signals cluster
        3: {RelationType.WENT_QUIET, RelationType.DECLINED, RelationType.HESITANT,
            RelationType.CANCELLED, RelationType.TRIAL_NEGATIVE},
        # Communication cluster
        4: {RelationType.SENT, RelationType.PROPOSAL_SENT, RelationType.MENTIONED,
            RelationType.EXPRESSED},
        # Organizational cluster
        5: {RelationType.REPORTS_TO, RelationType.EXPANDING, RelationType.ESCALATED},
        # Technical cluster
        6: {RelationType.INTEGRATED, RelationType.BREAKING_DOWN, RelationType.COMPARED},
        # Decision cluster
        7: {RelationType.APPROVED, RelationType.REJECTED, RelationType.EVALUATES,
            RelationType.CONFIRMS},
        # Competitive cluster
        8: {RelationType.LAUNCHED, RelationType.COMPETING, RelationType.COMPARED},
    }

    def relation_similarity(self, code_a: int, code_b: int) -> float:
        """
        Semantic similarity between two relation codes.
        Same code = 1.0, same cluster = 0.7, different cluster = 0.0.
        """
        if code_a == code_b:
            return 1.0
        for cluster in self._RELATION_CLUSTERS.values():
            if code_a in cluster and code_b in cluster:
                return 0.7
        return 0.0

    def modifier_similarity(self, code_a: int, code_b: int) -> float:
        """Semantic similarity between modifier codes."""
        if code_a == code_b:
            return 1.0
        # Sentiment-aligned modifiers are related
        negative = {Modifier.NEGATIVE, Modifier.URGENT, Modifier.ESCALATION, Modifier.COMPETITIVE}
        positive = {Modifier.POSITIVE, Modifier.RESOLUTION, Modifier.HIGH_VALUE}
        if (code_a in negative and code_b in negative) or (code_a in positive and code_b in positive):
            return 0.6
        return 0.0


def make_codebook_strand(
    entity_slots: list[tuple[int, str]],
    relation: int,
    modifier: int,
    temporal: int,
    domain: int,
    sentiment: int,
    confidence: int,
    timestamp: int,
    raw_hash: str,
) -> CodebookStrand:
    """Factory function for creating a new CodebookStrand with a UUID."""
    return CodebookStrand(
        strand_id=str(uuid.uuid4()),
        entity_slots=entity_slots,
        relation=relation,
        modifier=modifier,
        temporal=temporal,
        domain=domain,
        sentiment=max(-2, min(2, sentiment)),
        confidence=max(1, min(5, confidence)),
        timestamp=timestamp,
        raw_hash=raw_hash,
    )
