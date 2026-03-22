"""
MemorySystem — Unified interface for the Cognitive Memory Architecture (v2)

Ties all three layers together:
  Layer 1 — DNA Encoder:  raw text → CodebookStrand (finite alphabet)
  Layer 2 — Association Graph:  strand relationships + ego nodes + recency
  Layer 3 — Expression Engine:  spreading activation → DNA/RNA/Protein decode

New in v2:
  - Codebook-constrained encoding (real compression)
  - Entity instance registry (normalized cross-strand linking)
  - Ego nodes (agent identity anchoring)
  - Multi-resolution decode (DNA → RNA → Protein)
  - Recency priming (recently activated paths stay warm)
"""

from __future__ import annotations

import hashlib

from codebook import Codebook, CodebookStrand, Modifier
from entities import EntityRegistry
from genome import DNAEncoder, Genome
from graph import AssociationGraph
from expression import ExpressionEngine


class MemorySystem:
    """Three-layer cognitive memory architecture with codebook encoding."""

    def __init__(
        self,
        genome_path: str = "genome.json",
        graph_path: str = "graph.json",
        entities_path: str = "entities.json",
        model: str = "claude-sonnet-4-20250514",
    ):
        self.codebook = Codebook()
        self.entity_registry = EntityRegistry(path=entities_path)
        self.encoder = DNAEncoder(
            codebook=self.codebook,
            entity_registry=self.entity_registry,
            model=model,
        )
        self.genome = Genome(path=genome_path)
        self.graph = AssociationGraph(path=graph_path)
        self.expression = ExpressionEngine(
            genome=self.genome,
            graph=self.graph,
            codebook=self.codebook,
            entity_registry=self.entity_registry,
            model=model,
        )
        self._recent_ids: list[str] = list(self.genome.all_ids())

        # Ensure ego node exists
        self.graph.ensure_ego_node("agent")

    def store(self, raw_text: str, timestamp: int | None = None) -> CodebookStrand | None:
        """
        Encode and store a raw interaction.

        1. Dedup check via raw text hash
        2. Encode to CodebookStrand (codebook-constrained)
        3. Store in genome
        4. Register in association graph with entity-based edges
        5. Auto-link to ego node if personally significant

        Returns the created strand, or None if duplicate.
        """
        raw_hash = hashlib.sha256(raw_text.encode()).hexdigest()
        if self.genome.has_hash(raw_hash):
            return None

        # Layer 1: Encode
        strand = self.encoder.encode(raw_text, timestamp=timestamp)

        # Store in genome
        self.genome.add(strand)

        # Layer 2: Add to graph with entity-registry edges
        self.graph.add_strand(
            strand,
            recent_ids=self._recent_ids,
            entity_registry=self.entity_registry,
            genome_getter=self.genome.get,
        )

        # Auto-link to ego if urgent, deadline, or escalation
        if strand.modifier in (
            Modifier.URGENT.value,
            Modifier.DEADLINE.value,
            Modifier.ESCALATION.value,
        ):
            self.graph.link_to_ego(strand.strand_id, "agent")

        self._recent_ids.append(strand.strand_id)
        return strand

    def query(self, query_text: str) -> dict:
        """
        Query the memory system via spreading activation + multi-resolution decode.

        Returns expression result with activated memories, scores, decode levels,
        token costs, and API call savings.
        """
        query_strand = self.encoder.encode(query_text)
        return self.expression.express(query_strand)

    def stats(self) -> dict:
        """Return system statistics."""
        return {
            "total_strands": self.genome.count(),
            "graph_nodes": self.graph.node_count(),
            "graph_edges": self.graph.edge_count(),
            "entity_instances": self.entity_registry.count(),
            "ego_nodes": self.graph.ego_node_count(),
            "codebook_size": self.codebook.total_codes(),
        }
