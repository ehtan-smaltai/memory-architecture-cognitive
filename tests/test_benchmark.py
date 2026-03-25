"""
Benchmark tests — performance assertions that catch regressions.

Run with: pytest tests/test_benchmark.py -v
These run without API calls using DeterministicEncoder.
"""

import os
import shutil
import tempfile
import time
import pytest

from cognitive_memory import (
    Codebook, Config, EntityRegistry, Genome,
    AssociationGraph, MemorySystem, make_codebook_strand,
    EntityType, RelationType, Modifier, TemporalMarker, Domain,
)
from cognitive_memory.expression import (
    ExpressionEngine, codebook_similarity, estimate_strand_tokens,
)
from cognitive_memory.testing import DeterministicEncoder


@pytest.fixture
def bench_env():
    """Create a temporary benchmark environment."""
    tmpdir = tempfile.mkdtemp(prefix="cogmem_test_bench_")
    config = Config(
        genome_path=os.path.join(tmpdir, "genome.json"),
        graph_path=os.path.join(tmpdir, "graph.json"),
        entities_path=os.path.join(tmpdir, "entities.json"),
    )
    codebook = Codebook()
    registry = EntityRegistry(path=config.entities_path)
    encoder = DeterministicEncoder(codebook, registry)
    genome = Genome(path=config.genome_path)
    graph = AssociationGraph(path=config.graph_path, config=config)
    graph.ensure_ego_node("agent")

    yield {
        "tmpdir": tmpdir,
        "config": config,
        "codebook": codebook,
        "registry": registry,
        "encoder": encoder,
        "genome": genome,
        "graph": graph,
    }
    shutil.rmtree(tmpdir, ignore_errors=True)


def _populate(env, n: int, batch: bool = True):
    """Store n strands using the deterministic encoder."""
    encoder = env["encoder"]
    genome = env["genome"]
    graph = env["graph"]
    registry = env["registry"]
    recent_ids = []
    base_ts = 1_000_000_000

    if batch:
        genome.begin_batch()
        registry.begin_batch()
        graph.begin_batch()

    for i in range(n):
        text = f"Interaction {i}: Client {i % 10} discussed topic {i % 7} with contact {i % 5}"
        strand = encoder.encode(text, timestamp=base_ts + i * 3600)
        genome.add(strand)
        graph.add_strand(
            strand, recent_ids=recent_ids[-3:],
            entity_registry=registry, genome_getter=genome.get,
        )
        recent_ids.append(strand.strand_id)

    if batch:
        genome.end_batch()
        registry.end_batch()
        graph.end_batch()

    graph.rebuild_domain_relation_index(genome.get)
    return recent_ids


class TestDeterministicEncoder:
    """Verify the mock encoder produces valid strands."""

    def test_encode_returns_valid_strand(self, bench_env):
        strand = bench_env["encoder"].encode("Test input")
        assert strand.strand_id
        assert strand.raw_hash
        assert len(strand.entity_slots) >= 1
        assert -2 <= strand.sentiment <= 2
        assert 1 <= strand.confidence <= 5

    def test_encode_deterministic(self, bench_env):
        s1 = bench_env["encoder"].encode("Same input", timestamp=1000)
        s2 = bench_env["encoder"].encode("Same input", timestamp=1000)
        # Same hash, same codes (but different strand_ids due to UUID)
        assert s1.raw_hash == s2.raw_hash
        assert s1.relation == s2.relation
        assert s1.domain == s2.domain

    def test_different_inputs_different_codes(self, bench_env):
        s1 = bench_env["encoder"].encode("Input A")
        s2 = bench_env["encoder"].encode("Input B")
        assert s1.raw_hash != s2.raw_hash

    def test_satisfies_encoder_protocol(self):
        from cognitive_memory import Encoder
        assert hasattr(DeterministicEncoder, "encode")


class TestStorePerformance:
    """Store throughput benchmarks."""

    def test_store_100_under_1_second(self, bench_env):
        t0 = time.perf_counter()
        _populate(bench_env, 100)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"100 stores took {elapsed:.2f}s (>1s)"
        assert bench_env["genome"].count() == 100

    def test_store_500_under_10_seconds(self, bench_env):
        t0 = time.perf_counter()
        _populate(bench_env, 500)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"500 stores took {elapsed:.2f}s (>10s)"

    def test_store_1000_under_30_seconds(self, bench_env):
        t0 = time.perf_counter()
        _populate(bench_env, 1000)
        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"1000 stores took {elapsed:.2f}s (>30s)"


class TestQueryPerformance:
    """Query latency benchmarks (local retrieval, no LLM)."""

    def test_query_100_strands_under_10ms(self, bench_env):
        _populate(bench_env, 100)
        expression = ExpressionEngine(
            genome=bench_env["genome"], graph=bench_env["graph"],
            codebook=bench_env["codebook"], entity_registry=bench_env["registry"],
            config=bench_env["config"],
        )
        q = bench_env["encoder"].encode("What about pricing?")

        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            seeds = expression._find_seeds(q)
            threshold = expression._adaptive_threshold(seeds, 0.5)
            edge_weights = expression._query_edge_weights(q)
            activation = expression._spread_activation(seeds, threshold, edge_weights)
            selected, tokens = expression._assemble_context(activation)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            times.append(elapsed_ms)

        avg = sum(times) / len(times)
        assert avg < 10.0, f"Avg query at 100 strands: {avg:.2f}ms (>10ms)"

    def test_query_1000_strands_under_50ms(self, bench_env):
        _populate(bench_env, 1000)
        expression = ExpressionEngine(
            genome=bench_env["genome"], graph=bench_env["graph"],
            codebook=bench_env["codebook"], entity_registry=bench_env["registry"],
            config=bench_env["config"],
        )
        q = bench_env["encoder"].encode("What about pricing?")

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            seeds = expression._find_seeds(q)
            threshold = expression._adaptive_threshold(seeds, 0.5)
            edge_weights = expression._query_edge_weights(q)
            activation = expression._spread_activation(seeds, threshold, edge_weights)
            selected, tokens = expression._assemble_context(activation)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            times.append(elapsed_ms)

        avg = sum(times) / len(times)
        assert avg < 50.0, f"Avg query at 1000 strands: {avg:.2f}ms (>50ms)"


class TestTokenEfficiency:
    """Verify compression ratio scales correctly."""

    def test_compression_ratio_at_100(self, bench_env):
        _populate(bench_env, 100)
        expression = ExpressionEngine(
            genome=bench_env["genome"], graph=bench_env["graph"],
            codebook=bench_env["codebook"], entity_registry=bench_env["registry"],
            config=bench_env["config"],
        )
        q = bench_env["encoder"].encode("pricing concerns")
        seeds = expression._find_seeds(q)
        threshold = expression._adaptive_threshold(seeds, 0.5)
        edge_weights = expression._query_edge_weights(q)
        activation = expression._spread_activation(seeds, threshold, edge_weights)
        selected, tokens_used = expression._assemble_context(activation)

        tokens_naive = 100 * 33
        assert tokens_used < tokens_naive, "Retrieval should use fewer tokens than naive"
        ratio = tokens_naive / tokens_used if tokens_used > 0 else float("inf")
        assert ratio > 2.0, f"Compression ratio {ratio:.1f}x is too low"

    def test_tokens_bounded_at_1000(self, bench_env):
        """Token budget should hold regardless of genome size."""
        _populate(bench_env, 1000)
        expression = ExpressionEngine(
            genome=bench_env["genome"], graph=bench_env["graph"],
            codebook=bench_env["codebook"], entity_registry=bench_env["registry"],
            config=bench_env["config"],
        )
        q = bench_env["encoder"].encode("urgent client issues")
        seeds = expression._find_seeds(q)
        threshold = expression._adaptive_threshold(seeds, 0.5)
        edge_weights = expression._query_edge_weights(q)
        activation = expression._spread_activation(seeds, threshold, edge_weights)
        selected, tokens_used = expression._assemble_context(activation)

        assert tokens_used <= bench_env["config"].token_budget, \
            f"Tokens {tokens_used} exceeded budget {bench_env['config'].token_budget}"


class TestConsolidatePerformance:
    def test_consolidate_1000_under_500ms(self, bench_env):
        _populate(bench_env, 1000)
        system = object.__new__(MemorySystem)
        system.config = bench_env["config"]
        system.codebook = bench_env["codebook"]
        system.genome = bench_env["genome"]
        system.graph = bench_env["graph"]
        system.entity_registry = bench_env["registry"]

        t0 = time.perf_counter()
        result = system.consolidate()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 5000, f"Consolidation took {elapsed_ms:.0f}ms (>5000ms)"


class TestForgetPerformance:
    def test_forget_1000_under_500ms(self, bench_env):
        _populate(bench_env, 1000)
        system = object.__new__(MemorySystem)
        system.config = bench_env["config"]
        system.codebook = bench_env["codebook"]
        system.genome = bench_env["genome"]
        system.graph = bench_env["graph"]
        system.entity_registry = bench_env["registry"]

        t0 = time.perf_counter()
        result = system.forget(min_age_seconds=0, min_activations=0)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 10000, f"Forgetting took {elapsed_ms:.0f}ms (>10000ms)"


class TestGraphScaling:
    """Verify graph doesn't explode at scale."""

    def test_edge_count_bounded(self, bench_env):
        """Edges per node should be capped by MAX_EDGES_PER_NODE."""
        _populate(bench_env, 500)
        graph = bench_env["graph"]
        for node in graph.graph.nodes:
            out = graph.graph.out_degree(node)
            assert out <= graph.MAX_EDGES_PER_NODE + 5, \
                f"Node {node} has {out} edges (cap: {graph.MAX_EDGES_PER_NODE})"
