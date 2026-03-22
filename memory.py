"""
MemorySystem — Unified Interface for the Cognitive Memory Architecture

Combines two biological systems:

  MOLECULAR BIOLOGY (Storage):
    Protein (raw interaction) → RNA (transcription/extraction) → DNA (codebook compression)

  NEUROSCIENCE (Retrieval):
    Query → encode → spreading activation (local) → LLM reads DNA directly → answer

The key insight: DNA sits at the intersection of both systems.
It is both the OUTPUT of the molecular storage pipeline and the
INPUT to the brain-like retrieval pipeline.

API calls per operation:
  store():  1 call (RNA transcription — extract + compress to DNA)
  query():  2 calls (1 to encode query to DNA + 1 for brain-like reasoning)
"""

from __future__ import annotations

import hashlib

from codebook import Codebook, CodebookStrand, Modifier
from entities import EntityRegistry
from genome import DNAEncoder, Genome
from graph import AssociationGraph
from expression import ExpressionEngine


class MemorySystem:
    """
    Two-system cognitive memory:
      Storage  — Molecular biology (Protein → RNA → DNA)
      Retrieval — Neuroscience (spreading activation + direct DNA reasoning)
    """

    def __init__(
        self,
        genome_path: str = "genome.json",
        graph_path: str = "graph.json",
        entities_path: str = "entities.json",
        model: str = "claude-sonnet-4-20250514",
    ):
        self.codebook = Codebook()
        self.entity_registry = EntityRegistry(path=entities_path)

        # The RNA transcription layer (encodes protein → DNA)
        self.encoder = DNAEncoder(
            codebook=self.codebook,
            entity_registry=self.entity_registry,
            model=model,
        )

        # The genome (DNA storage)
        self.genome = Genome(path=genome_path)

        # The association graph (neural connections)
        self.graph = AssociationGraph(path=graph_path)

        # The brain (retrieval via direct DNA reasoning)
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
        MOLECULAR BIOLOGY PIPELINE: Protein → RNA → DNA

        1. Protein: raw_text IS the protein (the functional interaction)
        2. RNA transcription: Claude API extracts structured fields (1 API call)
        3. DNA compression: map to codebook integer sequence
        4. Store in genome.json
        5. Register in association graph
        6. Auto-link to ego node if significant

        Returns the DNA strand, or None if duplicate.
        """
        # Dedup check
        raw_hash = hashlib.sha256(raw_text.encode()).hexdigest()
        if self.genome.has_hash(raw_hash):
            return None

        # RNA transcription → DNA compression (1 API call)
        strand = self.encoder.encode(raw_text, timestamp=timestamp)

        # Store DNA in genome
        self.genome.add(strand)

        # Build neural connections in association graph
        self.graph.add_strand(
            strand,
            recent_ids=self._recent_ids,
            entity_registry=self.entity_registry,
            genome_getter=self.genome.get,
        )

        # Auto-link to ego if urgent/deadline/escalation
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
        NEUROSCIENCE PIPELINE: Brain-like retrieval

        1. Encode query to DNA codes (1 API call)
        2. Spreading activation through graph (LOCAL, 0 API calls)
        3. Budget selection within token limit (LOCAL)
        4. Feed activated DNA codes directly to LLM (1 API call)

        Total: 2 API calls. ~200-400 context tokens. Fixed regardless of genome size.

        Returns:
            {
                "answer": str,           # LLM's response
                "activated": [...],      # (strand_id, score, dna_code)
                "not_activated": [...],  # strand_ids not activated
                "tokens_used": int,
                "tokens_naive": int,
                "api_calls": int,        # always 1 (+ 1 for encoding = 2 total)
            }
        """
        # Encode query to DNA (1 API call — the RNA transcription of the query)
        query_strand = self.encoder.encode(query_text)

        # Brain-like retrieval (1 API call for reasoning)
        return self.expression.express(query_text, query_strand)

    def stats(self) -> dict:
        """System statistics."""
        return {
            "total_strands": self.genome.count(),
            "graph_nodes": self.graph.node_count(),
            "graph_edges": self.graph.edge_count(),
            "entity_instances": self.entity_registry.count(),
            "ego_nodes": self.graph.ego_node_count(),
            "codebook_size": self.codebook.total_codes(),
        }
