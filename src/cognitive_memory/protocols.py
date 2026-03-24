"""
Protocols — Abstract interfaces for pluggable components.

Defines the contracts that storage backends and encoders must satisfy,
enabling testing without API calls and swapping implementations.
"""

from __future__ import annotations

from typing import Protocol, Optional, runtime_checkable

from .codebook import CodebookStrand


@runtime_checkable
class Encoder(Protocol):
    """Protocol for encoding raw text into CodebookStrand units.

    The default implementation (DNAEncoder) calls the Claude API.
    Implement this protocol to use a different LLM, a local model,
    or a deterministic encoder for testing.
    """

    def encode(self, raw_text: str, timestamp: Optional[int] = None) -> CodebookStrand:
        """Encode raw text into a CodebookStrand."""
        ...


@runtime_checkable
class StrandStore(Protocol):
    """Protocol for persistent strand storage.

    The default implementation (Genome) uses a JSON file.
    Implement this protocol to back the genome with a database,
    object store, or any other persistence layer.
    """

    def add(self, strand: CodebookStrand) -> str: ...
    def get(self, strand_id: str) -> Optional[CodebookStrand]: ...
    def remove(self, strand_id: str) -> None: ...
    def has_hash(self, raw_hash: str) -> bool: ...
    def all_ids(self) -> list[str]: ...
    def all_strands(self) -> list[CodebookStrand]: ...
    def active_strands(self) -> list[CodebookStrand]: ...
    def active_ids(self) -> list[str]: ...
    def count(self) -> int: ...
    def save(self) -> None: ...
    def begin_batch(self) -> None: ...
    def end_batch(self) -> None: ...
    def increment_activation(self, strand_id: str) -> None: ...
    def supersede(self, old_id: str, new_id: str) -> None: ...
