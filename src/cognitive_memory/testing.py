"""
DeterministicEncoder — API-free encoder for benchmarking and testing.

Produces realistic CodebookStrand objects by hashing the input text to
deterministically select codebook codes. No network calls, no API key needed.

Usage::

    from cognitive_memory import Codebook, EntityRegistry
    from cognitive_memory.testing import DeterministicEncoder

    encoder = DeterministicEncoder(Codebook(), EntityRegistry())
    strand = encoder.encode("Client Acme has budget concerns")
"""

from __future__ import annotations

import hashlib
import time
from typing import Optional

from .codebook import (
    Codebook,
    CodebookStrand,
    EntityType,
    RelationType,
    Modifier,
    TemporalMarker,
    Domain,
    make_codebook_strand,
)
from .entities import EntityRegistry


# Pools for deterministic entity name generation
_PERSON_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank",
    "Iris", "James", "Karen", "Leo", "Maria", "Nate", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
]
_ORG_NAMES = [
    "Acme Corp", "Beta Ltd", "Gamma Inc", "Delta Co", "Epsilon Group",
    "Zeta Systems", "Eta Partners", "Theta Labs", "Iota Ventures", "Kappa Tech",
    "Lambda Solutions", "Mu Industries", "Nu Digital", "Xi Global", "Omicron AI",
]
_PRODUCT_NAMES = [
    "Salesforce", "Slack", "HubSpot", "Jira", "Notion",
    "Zoom", "Teams", "Asana", "Monday", "Trello",
]


class DeterministicEncoder:
    """Deterministic encoder for benchmarking — no API calls.

    Hashes the input text to deterministically pick codebook codes,
    entity names, and trace text. Produces realistic strand distributions.
    """

    def __init__(
        self,
        codebook: Codebook,
        entity_registry: EntityRegistry,
        max_entities: int = 4,
    ):
        self.codebook = codebook
        self.entity_registry = entity_registry
        self._max_entities = max_entities

        # Pre-compute code lists for deterministic selection
        self._entity_types = list(EntityType)
        self._relations = list(RelationType)
        self._modifiers = list(Modifier)
        self._temporals = list(TemporalMarker)
        self._domains = list(Domain)

    def encode(self, raw_text: str, timestamp: Optional[int] = None) -> CodebookStrand:
        """Encode raw text into a CodebookStrand deterministically."""
        if timestamp is None:
            timestamp = int(time.time())

        raw_hash = hashlib.sha256(raw_text.encode()).hexdigest()
        h = int(raw_hash, 16)

        # Deterministic code selection via hash bits
        relation = self._relations[h % len(self._relations)].value
        h >>= 8
        modifier = self._modifiers[h % len(self._modifiers)].value
        h >>= 8
        temporal = self._temporals[h % len(self._temporals)].value
        h >>= 8
        domain = self._domains[h % len(self._domains)].value
        h >>= 8
        sentiment = (h % 5) - 2  # -2 to +2
        h >>= 8
        confidence = (h % 5) + 1  # 1 to 5
        h >>= 8

        # Determine number of entities (1-3, weighted toward 2)
        n_entities = min(self._max_entities, max(1, (h % 4)))
        h >>= 4

        # Create a temporary strand_id for entity resolution
        temp_strand = make_codebook_strand(
            entity_slots=[], relation=0, modifier=0, temporal=0, domain=0,
            sentiment=0, confidence=3, timestamp=timestamp, raw_hash=raw_hash,
        )
        strand_id = temp_strand.strand_id

        # Pick entities deterministically
        entity_slots = []
        for i in range(n_entities):
            etype = self._entity_types[h % len(self._entity_types)]
            h >>= 4
            name = self._pick_name(etype, h)
            h >>= 8
            etype_code = etype.value
            instance_id = self.entity_registry.resolve(
                raw_name=name,
                entity_type=etype_code,
                strand_id=strand_id,
                timestamp=timestamp,
            )
            entity_slots.append((etype_code, instance_id))

        # Generate a deterministic trace
        trace = self._make_trace(raw_text, h)

        return CodebookStrand(
            strand_id=strand_id,
            entity_slots=entity_slots,
            relation=relation,
            modifier=modifier,
            temporal=temporal,
            domain=domain,
            sentiment=sentiment,
            confidence=confidence,
            timestamp=timestamp,
            raw_hash=raw_hash,
            trace=trace,
        )

    def _pick_name(self, etype: EntityType, h: int) -> str:
        if etype == EntityType.PERSON:
            return _PERSON_NAMES[h % len(_PERSON_NAMES)]
        if etype == EntityType.ORG:
            return _ORG_NAMES[h % len(_ORG_NAMES)]
        if etype in (EntityType.PRODUCT, EntityType.TOOL):
            return _PRODUCT_NAMES[h % len(_PRODUCT_NAMES)]
        return _ORG_NAMES[h % len(_ORG_NAMES)]

    def _make_trace(self, raw_text: str, h: int) -> str:
        words = raw_text.split()
        if len(words) <= 15:
            return raw_text
        # Take first 12 words as a compressed summary
        return " ".join(words[:12]) + "..."
