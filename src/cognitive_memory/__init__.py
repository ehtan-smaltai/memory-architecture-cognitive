"""
cognitive_memory — Two-pipeline cognitive memory architecture for AI agents.

Codebook compression + spreading activation retrieval, inspired by
molecular biology (storage) and neuroscience (retrieval).

Quick start::

    from cognitive_memory import MemorySystem, Config

    config = Config(model="claude-sonnet-4-20250514")
    mem = MemorySystem(config=config)

    mem.store("Client Acme Corp has budget concerns for Q3")
    result = mem.query("What's happening with Acme Corp?")
    print(result["answer"])
"""

from .config import Config
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
from .entities import EntityInstance, EntityRegistry
from .genome import DNAEncoder, Genome
from .graph import AssociationGraph
from .expression import ExpressionEngine, codebook_similarity, estimate_strand_tokens
from .memory import MemorySystem
from .protocols import Encoder, StrandStore

__all__ = [
    # Core
    "Config",
    "MemorySystem",
    # Codebook
    "Codebook",
    "CodebookStrand",
    "EntityType",
    "RelationType",
    "Modifier",
    "TemporalMarker",
    "Domain",
    "make_codebook_strand",
    # Entities
    "EntityInstance",
    "EntityRegistry",
    # Genome
    "DNAEncoder",
    "Genome",
    # Graph
    "AssociationGraph",
    # Expression
    "ExpressionEngine",
    "codebook_similarity",
    "estimate_strand_tokens",
    # Protocols
    "Encoder",
    "StrandStore",
]
