"""Tests for protocols — verify that concrete classes satisfy protocol contracts."""

import pytest
from cognitive_memory import Encoder, StrandStore, DNAEncoder, Genome


class TestProtocolConformance:
    def test_genome_is_strand_store(self):
        """Genome should satisfy the StrandStore protocol."""
        assert issubclass(Genome, StrandStore) or isinstance(Genome.__new__(Genome), StrandStore)
        # Check via runtime_checkable
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        os.unlink(tmp.name)
        try:
            g = Genome(path=tmp.name)
            assert isinstance(g, StrandStore)
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    def test_dna_encoder_is_encoder(self):
        """DNAEncoder should satisfy the Encoder protocol."""
        # We can't instantiate without API key, but we can check the class
        assert hasattr(DNAEncoder, "encode")
        # The encode method should have the right signature
        import inspect
        sig = inspect.signature(DNAEncoder.encode)
        params = list(sig.parameters.keys())
        assert "raw_text" in params
        assert "timestamp" in params
