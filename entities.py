"""
Entity Registry — Normalizes entity mentions across memory strands.

Solves the problem where "Sarah", "Acme's CTO", and "Sarah Chen" should all
refer to the same entity instance. Uses fuzzy substring matching + entity type
filtering to resolve mentions to canonical instance IDs.

Persists to entities.json. Tracks aliases, first/last seen timestamps,
and all referencing strand_ids per entity.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class EntityInstance:
    """A single normalized entity in the universe."""
    instance_id: str
    entity_type: int          # EntityType code from codebook
    canonical_name: str       # e.g., "Sarah Chen"
    aliases: set[str]         # e.g., {"sarah", "acme's cto", "sarah chen"}
    strand_ids: list[str]     # all strands referencing this entity
    first_seen: int           # timestamp
    last_seen: int            # timestamp

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "entity_type": self.entity_type,
            "canonical_name": self.canonical_name,
            "aliases": sorted(self.aliases),
            "strand_ids": self.strand_ids,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EntityInstance:
        return cls(
            instance_id=d["instance_id"],
            entity_type=d["entity_type"],
            canonical_name=d["canonical_name"],
            aliases=set(d["aliases"]),
            strand_ids=d["strand_ids"],
            first_seen=d["first_seen"],
            last_seen=d["last_seen"],
        )


class EntityRegistry:
    """
    Normalizes entity mentions across strands.

    When encoding a new strand, call resolve() with each entity mention.
    The registry will either match it to an existing instance (via alias
    lookup or fuzzy matching) or create a new one.
    """

    def __init__(self, path: str = "entities.json"):
        self.path = path
        self._entities: dict[str, EntityInstance] = {}
        self._alias_index: dict[str, str] = {}  # normalized_alias → instance_id
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
            for d in data:
                inst = EntityInstance.from_dict(d)
                self._entities[inst.instance_id] = inst
                for alias in inst.aliases:
                    self._alias_index[alias] = inst.instance_id

    def save(self):
        data = [inst.to_dict() for inst in self._entities.values()]
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def resolve(
        self,
        raw_name: str,
        entity_type: int,
        strand_id: str,
        timestamp: int,
    ) -> str:
        """
        Resolve a raw entity mention to a canonical instance_id.

        Resolution order:
          1. Exact alias match
          2. Fuzzy match (substring containment + same entity type)
          3. Create new instance

        Returns the instance_id.
        """
        normalized = raw_name.lower().strip()

        # 1. Exact alias match
        if normalized in self._alias_index:
            inst_id = self._alias_index[normalized]
            inst = self._entities[inst_id]
            if strand_id not in inst.strand_ids:
                inst.strand_ids.append(strand_id)
            inst.last_seen = max(inst.last_seen, timestamp)
            self.save()
            return inst_id

        # 2. Fuzzy match: substring containment + same entity type
        for inst_id, inst in self._entities.items():
            if inst.entity_type != entity_type:
                continue
            for alias in inst.aliases:
                # Check if one contains the other (handles "Sarah" vs "Sarah Chen")
                if (len(normalized) >= 3 and normalized in alias) or (
                    len(alias) >= 3 and alias in normalized
                ):
                    # Match found — register new alias
                    inst.aliases.add(normalized)
                    self._alias_index[normalized] = inst_id
                    if strand_id not in inst.strand_ids:
                        inst.strand_ids.append(strand_id)
                    inst.last_seen = max(inst.last_seen, timestamp)
                    self.save()
                    return inst_id

        # 3. No match — create new instance
        inst_id = self._make_id(raw_name)
        # Avoid ID collision
        if inst_id in self._entities:
            inst_id = f"{inst_id}_{len(self._entities)}"

        self._entities[inst_id] = EntityInstance(
            instance_id=inst_id,
            entity_type=entity_type,
            canonical_name=raw_name,
            aliases={normalized},
            strand_ids=[strand_id],
            first_seen=timestamp,
            last_seen=timestamp,
        )
        self._alias_index[normalized] = inst_id
        self.save()
        return inst_id

    def get(self, instance_id: str) -> EntityInstance | None:
        """Get an entity instance by ID."""
        return self._entities.get(instance_id)

    def get_strands_for_entity(self, instance_id: str) -> list[str]:
        """All strand_ids referencing this entity."""
        inst = self._entities.get(instance_id)
        if inst is None:
            return []
        return inst.strand_ids

    def all_instances(self) -> list[EntityInstance]:
        """Return all entity instances."""
        return list(self._entities.values())

    def count(self) -> int:
        return len(self._entities)

    def _make_id(self, name: str) -> str:
        """Generate a stable, readable instance_id from a name."""
        return (
            name.lower()
            .strip()
            .replace(" ", "_")
            .replace("'", "")
            .replace(".", "")
            .replace(",", "")
            .replace("-", "_")
        )
