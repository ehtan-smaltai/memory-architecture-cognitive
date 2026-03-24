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
import logging
import os
import re
import time
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

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
  "confidence": 4,
  "trace": "10-15 word compressed summary preserving key facts"
}}

Rules:
- entities: 1-4 entities. "type" MUST be one of the ENTITY_TYPES codes above. "name" is the entity's actual name (e.g., "Sarah", "Acme Corp").
- relation: MUST be one of the RELATIONS codes above. Pick the closest match.
- modifier: MUST be one of the MODIFIERS codes above.
- temporal: MUST be one of the TEMPORAL codes above.
- domain: MUST be one of the DOMAINS codes above.
- sentiment: integer from -2 (very negative) to 2 (very positive). 0 = neutral.
- confidence: integer from 1 (very uncertain) to 5 (very certain).
- trace: a 10-15 word compressed micro-summary that preserves specific facts (numbers, durations, dates, names) that the codebook codes cannot capture. This is the neocortical memory trace.

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

    @staticmethod
    def _parse_llm_json(text: str) -> dict:
        """Parse JSON from LLM output, stripping markdown fences and fixing common issues."""
        text = text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0].strip()
        # Try to extract JSON object if surrounded by extra text
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
        return json.loads(text)

    def _extract_with_retry(self, raw_text: str, max_retries: int = 2) -> Optional[dict]:
        """Call LLM to extract structured fields, with retry on parse failure."""
        for attempt in range(max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    system=self._system_prompt,
                    messages=[{"role": "user", "content": raw_text}],
                )
                text = response.content[0].text.strip()
                extracted = self._parse_llm_json(text)

                # Validate required keys
                if "entities" not in extracted or not isinstance(extracted["entities"], list):
                    raise ValueError("Missing or invalid 'entities' field")
                if "relation" not in extracted:
                    raise ValueError("Missing 'relation' field")

                return extracted
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"LLM extraction attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    return None
            except anthropic.APIError as e:
                logger.error(f"API error during extraction: {e}")
                if attempt == max_retries:
                    return None
        return None

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

        extracted = self._extract_with_retry(raw_text, max_retries=2)

        if extracted is None:
            # Fallback: return a minimal strand with UNKNOWN/OTHER codes
            logger.warning("LLM extraction failed for input, using fallback encoding")
            extracted = {
                "entities": [{"type": "UNKNOWN", "name": "unknown"}],
                "relation": "OTHER",
                "modifier": "NEUTRAL",
                "temporal": "PRESENT",
                "domain": "GENERAL",
                "sentiment": 0,
                "confidence": 1,
                "trace": raw_text[:80],
            }

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

        # Extract neocortical trace (micro-summary preserving key facts)
        trace = extracted.get("trace", "")

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
            trace=trace,
        )

        return strand


# ─── Genome persistence ─────────────────────────────────────────────────────

class Genome:
    """Persistent store for CodebookStrand units (genome.json)."""

    def __init__(self, path: str = "genome.json"):
        self.path = path
        self._strands: dict[str, CodebookStrand] = {}
        self._batch_mode: bool = False
        self._load()

    def begin_batch(self):
        """Suppress auto-save until end_batch() is called."""
        self._batch_mode = True

    def end_batch(self):
        """End batch mode and persist."""
        self._batch_mode = False
        self.save()

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
        if not self._batch_mode:
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

    def active_strands(self) -> list[CodebookStrand]:
        """Return only non-superseded strands."""
        return [s for s in self._strands.values() if s.superseded_by is None]

    def active_ids(self) -> list[str]:
        """Return IDs of non-superseded strands."""
        return [s.strand_id for s in self._strands.values() if s.superseded_by is None]

    def remove(self, strand_id: str):
        """Remove a strand from the genome (forgetting)."""
        if strand_id in self._strands:
            del self._strands[strand_id]
            if not self._batch_mode:
                self.save()

    def increment_activation(self, strand_id: str):
        """Track that this strand was activated during a query."""
        strand = self._strands.get(strand_id)
        if strand:
            strand.activation_count += 1

    def supersede(self, old_id: str, new_id: str):
        """Mark a strand as superseded by another."""
        old = self._strands.get(old_id)
        if old:
            old.superseded_by = new_id
            if not self._batch_mode:
                self.save()
