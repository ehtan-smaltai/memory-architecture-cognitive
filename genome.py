"""
Genome — Molecular Biology Storage Pipeline

Implements the storage half of the architecture:
  Protein (raw interaction) → RNA (transcription) → DNA (codebook compression)

  - Protein: the raw text input IS the protein — the functional event
  - RNA transcription: Claude API extracts structured fields from raw text
  - DNA compression: map extracted fields to codebook integer codes

The output is a CodebookStrand — a fixed-width integer sequence stored
in genome.json. This is the DNA that the brain-like retrieval layer reads.
"""

from __future__ import annotations

import json
import hashlib
import os
import time
from typing import Optional

import anthropic

from codebook import (
    Codebook,
    CodebookStrand,
    make_codebook_strand,
    EntityType,
    RelationType,
    Modifier,
    TemporalMarker,
    Domain,
)
from entities import EntityRegistry


# ─── Extraction prompt (constrained to codebook codes) ───────────────────────

def build_extraction_prompt(codebook: Codebook) -> str:
    """Build the system prompt that forces Claude to use codebook codes."""
    entity_types = ", ".join(codebook.entity_type_list())
    relations = ", ".join(codebook.relation_list())
    modifiers = ", ".join(codebook.modifier_list())
    temporals = ", ".join(codebook.temporal_list())
    domains = ", ".join(codebook.domain_list())

    return f"""You are a memory encoder. Given raw text, classify it using ONLY these codebook codes.

ENTITY_TYPES: {entity_types}
RELATIONS: {relations}
MODIFIERS: {modifiers}
TEMPORAL: {temporals}
DOMAINS: {domains}

Return ONLY valid JSON with these exact keys:
{{
  "entities": [{{"type": "ENTITY_TYPE_CODE", "name": "entity_name"}}, ...],
  "relation": "RELATION_CODE",
  "modifier": "MODIFIER_CODE",
  "temporal": "TEMPORAL_CODE",
  "domain": "DOMAIN_CODE",
  "sentiment": 0,
  "confidence": 4
}}

Rules:
- entities: 1-4 entities. "type" MUST be one of the ENTITY_TYPES codes above. "name" is the entity's actual name (e.g., "Sarah", "Acme Corp").
- relation: MUST be one of the RELATIONS codes above. Pick the closest match.
- modifier: MUST be one of the MODIFIERS codes above.
- temporal: MUST be one of the TEMPORAL codes above.
- domain: MUST be one of the DOMAINS codes above.
- sentiment: integer from -2 (very negative) to 2 (very positive). 0 = neutral.
- confidence: integer from 1 (very uncertain) to 5 (very certain).

Return ONLY the JSON object, no markdown fences, no explanation."""


# ─── Encoder ─────────────────────────────────────────────────────────────────

class DNAEncoder:
    """Encodes raw text into CodebookStrand units via constrained Claude API."""

    def __init__(
        self,
        codebook: Codebook,
        entity_registry: EntityRegistry,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.codebook = codebook
        self.entity_registry = entity_registry
        self._system_prompt = build_extraction_prompt(codebook)

    def encode(self, raw_text: str, timestamp: Optional[int] = None) -> CodebookStrand:
        """
        Encode raw text into a CodebookStrand.

        1. Call Claude API with codebook-constrained prompt
        2. Map response to codebook integer codes
        3. Resolve entity names through EntityRegistry
        4. Return fixed-width CodebookStrand
        """
        if timestamp is None:
            timestamp = int(time.time())

        raw_hash = hashlib.sha256(raw_text.encode()).hexdigest()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=self._system_prompt,
            messages=[{"role": "user", "content": raw_text}],
        )

        text = response.content[0].text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0].strip()
        extracted = json.loads(text)

        # Create a temporary strand_id for entity resolution
        temp_strand = make_codebook_strand(
            entity_slots=[],
            relation=0,
            modifier=0,
            temporal=0,
            domain=0,
            sentiment=0,
            confidence=3,
            timestamp=timestamp,
            raw_hash=raw_hash,
        )
        strand_id = temp_strand.strand_id

        # Map entities through codebook + registry
        entity_slots = []
        for ent in extracted.get("entities", [])[:4]:  # max 4 entities
            etype_code = self.codebook.encode_entity_type(ent.get("type", "UNKNOWN"))
            instance_id = self.entity_registry.resolve(
                raw_name=ent.get("name", "unknown"),
                entity_type=etype_code,
                strand_id=strand_id,
                timestamp=timestamp,
            )
            entity_slots.append((etype_code, instance_id))

        # Map other fields through codebook
        relation = self.codebook.encode_relation(extracted.get("relation", "OTHER"))
        modifier = self.codebook.encode_modifier(extracted.get("modifier", "NEUTRAL"))
        temporal = self.codebook.encode_temporal(extracted.get("temporal", "PRESENT"))
        domain = self.codebook.encode_domain(extracted.get("domain", "GENERAL"))
        sentiment = max(-2, min(2, int(extracted.get("sentiment", 0))))
        confidence = max(1, min(5, int(extracted.get("confidence", 3))))

        # Build the final strand (reusing the same strand_id)
        strand = CodebookStrand(
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
        )

        return strand


# ─── Genome persistence ─────────────────────────────────────────────────────

class Genome:
    """Persistent store for CodebookStrand units (genome.json)."""

    def __init__(self, path: str = "genome.json"):
        self.path = path
        self._strands: dict[str, CodebookStrand] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
            for d in data:
                strand = CodebookStrand.from_dict(d)
                self._strands[strand.strand_id] = strand

    def save(self):
        data = [s.to_dict() for s in self._strands.values()]
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, strand: CodebookStrand) -> str:
        """Add a strand to the genome. Returns strand_id."""
        self._strands[strand.strand_id] = strand
        self.save()
        return strand.strand_id

    def get(self, strand_id: str) -> Optional[CodebookStrand]:
        """Retrieve a single strand by ID."""
        return self._strands.get(strand_id)

    def has_hash(self, raw_hash: str) -> bool:
        """Check for duplicate by raw text hash."""
        return any(s.raw_hash == raw_hash for s in self._strands.values())

    def all_ids(self) -> list[str]:
        return list(self._strands.keys())

    def count(self) -> int:
        return len(self._strands)

    def all_strands(self) -> list[CodebookStrand]:
        return list(self._strands.values())
