"""
MemorySystem — Unified interface tying all three layers together.

Usage:
    system = MemorySystem()
    system.store("Client mentioned budget concerns")
    result = system.query("What are the budget issues?")
"""

import hashlib
import time

from genome import DNAEncoder, Genome
from graph import AssociationGraph
from expression import ExpressionEngine


class MemorySystem:
    """
    Three-layer cognitive memory architecture:
      Layer 1 — DNA Encoder:  raw text → compressed strand
      Layer 2 — Association Graph:  strand relationships + Hebbian learning
      Layer 3 — Expression Engine:  spreading activation → selective decode
    """

    def __init__(
        self,
        genome_path: str = "genome.json",
        graph_path: str = "graph.json",
        model: str = "claude-sonnet-4-20250514",
    ):
        self.encoder = DNAEncoder(model=model)
        self.genome = Genome(path=genome_path)
        self.graph = AssociationGraph(path=graph_path)
        self.expression = ExpressionEngine(
            genome=self.genome,
            graph=self.graph,
            model=model,
        )
        self._recent_ids: list[str] = list(self.genome.all_ids())

    def store(self, raw_text: str, timestamp: int | None = None) -> dict:
        """
        Encode and store a raw interaction.

        Returns the created strand.
        """
        # Dedup check
        raw_hash = hashlib.sha256(raw_text.encode()).hexdigest()
        if self.genome.has_hash(raw_hash):
            print(f"  [DEDUP] Skipping duplicate: {raw_text[:50]}...")
            return {}

        # Layer 1: Encode
        strand = self.encoder.encode(raw_text, timestamp=timestamp)

        # Store in genome
        self.genome.add(strand)

        # Layer 2: Add to graph with auto-edges
        self.graph.add_strand(
            strand,
            recent_ids=self._recent_ids,
            all_strands_getter=self.genome.get,
        )

        self._recent_ids.append(strand["strand_id"])

        return strand

    def query(self, query_text: str) -> dict:
        """
        Query the memory system using spreading activation.

        Returns expression result with activated memories, scores, and token costs.
        """
        # Create a temporary query strand (not saved to genome)
        query_strand = self.encoder.encode(query_text)

        # Run expression pipeline
        return self.expression.express(query_strand)

    def stats(self) -> dict:
        """Return system statistics."""
        return {
            "total_strands": self.genome.count(),
            "graph_nodes": self.graph.node_count(),
            "graph_edges": self.graph.edge_count(),
        }
