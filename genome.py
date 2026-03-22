"""
DNA Encoder — Layer 1 of the Cognitive Memory Architecture

Compresses raw interactions into minimal encoded memory units ("strands").
Each strand encodes entities, relations, sentiment, confidence, domain —
analogous to how DNA encodes biological information in a minimal alphabet.

Key property: raw text is NEVER stored. Only the compressed strand + a hash
of the original for dedup.
"""

import json
import uuid
import hashlib
import os
from typing import Optional

import anthropic


# ─── Strand schema ───────────────────────────────────────────────────────────

def make_strand(
    entities: list[str],
    relation: str,
    value: str,
    sentiment: float,
    confidence: float,
    timestamp: int,
    domain: str,
    raw_hash: str,
) -> dict:
    """Create a new memory strand with a unique ID."""
    return {
        "strand_id": str(uuid.uuid4()),
        "encoded": {
            "entities": entities,
            "relation": relation,
            "value": value,
            "sentiment": round(sentiment, 2),
            "confidence": round(confidence, 2),
            "timestamp": timestamp,
            "domain": domain,
        },
        "raw_hash": raw_hash,
        "edges": [],
    }


# ─── Encoder ─────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a memory encoder. Given raw text, extract structured fields.
Return ONLY valid JSON with these exact keys:
{
  "entities": ["list", "of", "named_entities"],
  "relation": "verb_phrase_describing_action",
  "value": "object_or_outcome",
  "sentiment": 0.0,
  "confidence": 0.85,
  "domain": "category"
}

Rules:
- entities: proper nouns, people, companies, products (2-5 items)
- relation: a concise verb phrase (e.g., "expressed_concern_about", "requested", "signed_up_for")
- value: the object/outcome of the relation (1-3 words)
- sentiment: float -1.0 (very negative) to 1.0 (very positive), 0.0 = neutral
- confidence: how certain you are about the extraction, 0.0 to 1.0
- domain: one of: sales, technical, operations, hr, finance, general

Return ONLY the JSON object, no markdown, no explanation."""


class DNAEncoder:
    """Encodes raw text into compressed memory strands using Claude API."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model

    def encode(self, raw_text: str, timestamp: Optional[int] = None) -> dict:
        """Encode a raw text interaction into a memory strand."""
        import time

        if timestamp is None:
            timestamp = int(time.time())

        raw_hash = hashlib.sha256(raw_text.encode()).hexdigest()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": raw_text}],
        )

        text = response.content[0].text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        extracted = json.loads(text)

        return make_strand(
            entities=extracted["entities"],
            relation=extracted["relation"],
            value=extracted["value"],
            sentiment=float(extracted["sentiment"]),
            confidence=float(extracted["confidence"]),
            timestamp=timestamp,
            domain=extracted["domain"],
            raw_hash=raw_hash,
        )


# ─── Genome persistence ─────────────────────────────────────────────────────

class Genome:
    """Persistent store for memory strands (genome.json)."""

    def __init__(self, path: str = "genome.json"):
        self.path = path
        self._strands: dict[str, dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
            self._strands = {s["strand_id"]: s for s in data}

    def save(self):
        with open(self.path, "w") as f:
            json.dump(list(self._strands.values()), f, indent=2)

    def add(self, strand: dict) -> str:
        """Add a strand to the genome. Returns strand_id."""
        sid = strand["strand_id"]
        self._strands[sid] = strand
        self.save()
        return sid

    def get(self, strand_id: str) -> Optional[dict]:
        """Retrieve a single strand by ID — does NOT load the full genome."""
        return self._strands.get(strand_id)

    def has_hash(self, raw_hash: str) -> bool:
        """Check if a strand with this raw_hash already exists (dedup)."""
        return any(s["raw_hash"] == raw_hash for s in self._strands.values())

    def all_ids(self) -> list[str]:
        """Return all strand IDs without loading full strand data."""
        return list(self._strands.keys())

    def count(self) -> int:
        return len(self._strands)

    def all_strands(self) -> list[dict]:
        """Return all strands. Use sparingly — prefer get() by ID."""
        return list(self._strands.values())
