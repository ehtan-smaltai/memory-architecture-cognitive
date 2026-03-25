#!/usr/bin/env python3
"""
Benchmark — Scaling & Performance Suite

Measures local computation cost (no API calls) at various genome sizes.
Uses DeterministicEncoder to build realistic genomes, then exercises:

  - Store throughput (strands/sec)
  - Query latency (seed finding + spreading activation + context assembly)
  - Consolidation speed
  - Forgetting speed
  - Token efficiency (compression ratio vs naive)
  - Graph density metrics

Run:
    python -m benchmarks.bench_scaling          # quick (default)
    python -m benchmarks.bench_scaling --full   # 10k strands
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import tempfile
import time

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cognitive_memory import (
    Codebook,
    Config,
    EntityRegistry,
    Genome,
    AssociationGraph,
    MemorySystem,
    make_codebook_strand,
    EntityType,
    RelationType,
    Modifier,
    TemporalMarker,
    Domain,
)
from cognitive_memory.expression import (
    ExpressionEngine,
    codebook_similarity,
    estimate_strand_tokens,
)
from cognitive_memory.testing import DeterministicEncoder


# ─── Corpus generation ───────────────────────────────────────────────────────

INTERACTION_TEMPLATES = [
    "Client {org} mentioned they have budget concerns for Q{q}",
    "{person} from {org} prefers technical documentation over slide decks",
    "Sent {org} a proposal for ${amount}/month enterprise plan",
    "{org} asked about integration with {tool}",
    "Client {org} is expanding their team to {n} people",
    "{person} from {org} reports to VP of {dept}",
    "{org} expressed urgency — they need solution by end of month",
    "Had a call with {org} — {person} seemed hesitant on pricing",
    "{org} asked specifically about {tool} integration",
    "Competitor of {org} just launched a similar product",
    "{person} from {org} mentioned their current tool is breaking down",
    "{org} requested a discount — mentioned budget constraints again",
    "{org} signed up for a {n}-week trial",
    "{person} from {org} went quiet for {n} weeks after seeing pricing",
    "{org} trial is going well — {person} sent positive feedback",
    "{org} budget cycle resets in January — {person} confirmed",
    "{person} from {org} asked about annual pricing vs monthly",
    "Sent {org} a custom proposal with {tool} integration highlighted",
    "{org} re-engaged — {person} asked to revisit the proposal",
    "{person} from {org} asked for a reference call with existing customer",
]

PEOPLE = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank",
          "Iris", "James", "Karen", "Leo", "Maria", "Nate", "Olivia", "Paul"]
ORGS = ["Acme Corp", "Beta Ltd", "Gamma Inc", "Delta Co", "Epsilon Group",
        "Zeta Systems", "Eta Partners", "Theta Labs", "Iota Ventures", "Kappa Tech"]
TOOLS = ["Salesforce", "Slack", "HubSpot", "Jira", "Notion", "Zoom"]
DEPTS = ["Sales", "Engineering", "Ops", "Marketing", "Finance"]


def generate_corpus(n: int) -> list[str]:
    """Generate n unique interaction strings."""
    corpus = []
    for i in range(n):
        tpl = INTERACTION_TEMPLATES[i % len(INTERACTION_TEMPLATES)]
        text = tpl.format(
            org=ORGS[i % len(ORGS)],
            person=PEOPLE[i % len(PEOPLE)],
            tool=TOOLS[i % len(TOOLS)],
            dept=DEPTS[i % len(DEPTS)],
            amount=(i + 1) * 500,
            n=(i % 12) + 1,
            q=(i % 4) + 1,
        )
        # Make each unique by appending index
        corpus.append(f"{text} (interaction #{i})")
    return corpus


# ─── Query corpus ────────────────────────────────────────────────────────────

QUERY_TEMPLATES = [
    "What's the situation with {org}'s pricing concerns?",
    "{person} wants to move forward, what do I need to know?",
    "Which client is closer to closing?",
    "What do we know about {org}'s integration needs?",
    "Summarize {person}'s engagement history",
    "Any urgent items across all clients?",
    "What happened with {org}'s trial?",
    "Compare {org} and {org2} deal progress",
]


def generate_queries(n: int) -> list[str]:
    queries = []
    for i in range(n):
        tpl = QUERY_TEMPLATES[i % len(QUERY_TEMPLATES)]
        queries.append(tpl.format(
            org=ORGS[i % len(ORGS)],
            org2=ORGS[(i + 1) % len(ORGS)],
            person=PEOPLE[i % len(PEOPLE)],
        ))
    return queries


# ─── Timer context manager ──────────────────────────────────────────────────

class Timer:
    def __init__(self):
        self.elapsed = 0.0
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


# ─── Benchmark runner ───────────────────────────────────────────────────────

def run_benchmark(
    genome_size: int,
    n_queries: int = 10,
    label: str = "",
) -> dict:
    """Run a full benchmark at the given genome size."""
    tmpdir = tempfile.mkdtemp(prefix="cogmem_bench_")

    try:
        config = Config(
            genome_path=os.path.join(tmpdir, "genome.json"),
            graph_path=os.path.join(tmpdir, "graph.json"),
            entities_path=os.path.join(tmpdir, "entities.json"),
        )

        codebook = Codebook()
        entity_registry = EntityRegistry(path=config.entities_path)
        encoder = DeterministicEncoder(codebook, entity_registry)
        genome = Genome(path=config.genome_path)
        graph = AssociationGraph(path=config.graph_path, config=config)
        graph.ensure_ego_node("agent")

        # ── Phase 1: Store ────────────────────────────────────────────
        corpus = generate_corpus(genome_size)
        recent_ids: list[str] = []

        store_times: list[float] = []
        base_ts = 1_000_000_000

        genome.begin_batch()
        entity_registry.begin_batch()
        graph.begin_batch()

        for i, text in enumerate(corpus):
            ts = base_ts + i * 3600

            with Timer() as t:
                strand = encoder.encode(text, timestamp=ts)
                genome.add(strand)
                graph.add_strand(
                    strand,
                    recent_ids=recent_ids[-3:],
                    entity_registry=entity_registry,
                    genome_getter=genome.get,
                )
                recent_ids.append(strand.strand_id)

            store_times.append(t.elapsed)

        genome.end_batch()
        entity_registry.end_batch()
        graph.end_batch()

        graph.rebuild_domain_relation_index(genome.get)

        # ── Phase 2: Query (local only — no LLM reasoning call) ──────
        expression = ExpressionEngine(
            genome=genome,
            graph=graph,
            codebook=codebook,
            entity_registry=entity_registry,
            config=config,
        )

        queries = generate_queries(n_queries)
        query_times: list[float] = []
        tokens_used_list: list[int] = []
        seeds_found: list[int] = []
        strands_activated: list[int] = []

        for q_text in queries:
            # Encode query deterministically
            q_strand = encoder.encode(q_text, timestamp=base_ts + genome_size * 3600)

            with Timer() as t:
                # Run the retrieval pipeline (minus LLM reasoning)
                query_complexity = expression._estimate_query_complexity(q_strand)
                seed_list = expression._find_seeds(q_strand)
                threshold = expression._adaptive_threshold(seed_list, query_complexity)
                edge_weights = expression._query_edge_weights(q_strand)
                activation = expression._spread_activation(seed_list, threshold, edge_weights)
                selected, tokens_used = expression._assemble_context(activation)
                entity_groups = expression._group_by_entity(selected)

            query_times.append(t.elapsed)
            tokens_used_list.append(tokens_used)
            seeds_found.append(len(seed_list))
            strands_activated.append(len(selected))

        # ── Phase 3: Consolidation ───────────────────────────────────
        # Build a minimal MemorySystem for consolidation/forget
        system = object.__new__(MemorySystem)
        system.config = config
        system.codebook = codebook
        system.genome = genome
        system.graph = graph
        system.entity_registry = entity_registry

        with Timer() as t_consolidate:
            consolidate_result = system.consolidate()

        # ── Phase 4: Forgetting ──────────────────────────────────────
        with Timer() as t_forget:
            forget_result = system.forget(min_age_seconds=0, min_activations=1)

        # ── Metrics ──────────────────────────────────────────────────
        tokens_naive = genome_size * 33
        avg_tokens = statistics.mean(tokens_used_list) if tokens_used_list else 0
        compression = tokens_naive / avg_tokens if avg_tokens > 0 else float("inf")

        return {
            "genome_size": genome_size,
            "label": label,
            # Store
            "store_total_s": sum(store_times),
            "store_avg_ms": statistics.mean(store_times) * 1000,
            "store_p50_ms": statistics.median(store_times) * 1000,
            "store_p99_ms": sorted(store_times)[int(len(store_times) * 0.99)] * 1000 if store_times else 0,
            "store_throughput": genome_size / sum(store_times) if store_times else 0,
            # Query
            "query_avg_ms": statistics.mean(query_times) * 1000,
            "query_p50_ms": statistics.median(query_times) * 1000,
            "query_p99_ms": sorted(query_times)[int(len(query_times) * 0.99)] * 1000 if query_times else 0,
            "query_max_ms": max(query_times) * 1000 if query_times else 0,
            # Retrieval quality
            "avg_seeds": statistics.mean(seeds_found),
            "avg_activated": statistics.mean(strands_activated),
            "avg_tokens_used": avg_tokens,
            "tokens_naive": tokens_naive,
            "compression_ratio": compression,
            # Graph
            "graph_nodes": graph.node_count(),
            "graph_edges": graph.edge_count(),
            "entities": entity_registry.count(),
            # Consolidation
            "consolidate_ms": t_consolidate.elapsed * 1000,
            "consolidated": consolidate_result["consolidated"],
            # Forgetting
            "forget_ms": t_forget.elapsed * 1000,
            "forgotten": forget_result["forgotten"],
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── Output formatting ──────────────────────────────────────────────────────

def print_results(results: list[dict]):
    """Print benchmark results as a formatted table."""
    print()
    print("=" * 90)
    print("  COGNITIVE MEMORY BENCHMARK — Scaling & Performance")
    print("=" * 90)

    # Store performance
    print("\n  STORE THROUGHPUT")
    print("  " + "-" * 75)
    print(f"  {'Genome':>8} | {'Total(s)':>9} | {'Avg(ms)':>9} | {'P50(ms)':>9} | {'P99(ms)':>9} | {'strands/s':>10}")
    print("  " + "-" * 75)
    for r in results:
        print(f"  {r['genome_size']:>8,} | {r['store_total_s']:>9.2f} | "
              f"{r['store_avg_ms']:>9.3f} | {r['store_p50_ms']:>9.3f} | "
              f"{r['store_p99_ms']:>9.3f} | {r['store_throughput']:>10.0f}")

    # Query performance
    print("\n  QUERY LATENCY (local retrieval, no LLM call)")
    print("  " + "-" * 75)
    print(f"  {'Genome':>8} | {'Avg(ms)':>9} | {'P50(ms)':>9} | {'P99(ms)':>9} | {'Max(ms)':>9} | {'Seeds':>6} | {'Active':>7}")
    print("  " + "-" * 75)
    for r in results:
        print(f"  {r['genome_size']:>8,} | {r['query_avg_ms']:>9.3f} | "
              f"{r['query_p50_ms']:>9.3f} | {r['query_p99_ms']:>9.3f} | "
              f"{r['query_max_ms']:>9.3f} | {r['avg_seeds']:>6.1f} | {r['avg_activated']:>7.1f}")

    # Token efficiency
    print("\n  TOKEN EFFICIENCY")
    print("  " + "-" * 65)
    print(f"  {'Genome':>8} | {'Tokens Used':>12} | {'Naive':>12} | {'Compression':>12} | {'Savings':>8}")
    print("  " + "-" * 65)
    for r in results:
        savings = (1 - 1/r['compression_ratio']) * 100 if r['compression_ratio'] != float('inf') else 100
        print(f"  {r['genome_size']:>8,} | {r['avg_tokens_used']:>12.0f} | "
              f"{r['tokens_naive']:>12,} | {r['compression_ratio']:>11.1f}x | "
              f"{savings:>7.1f}%")

    # Graph density
    print("\n  GRAPH DENSITY")
    print("  " + "-" * 55)
    print(f"  {'Genome':>8} | {'Nodes':>8} | {'Edges':>8} | {'Entities':>8} | {'Edges/Node':>10}")
    print("  " + "-" * 55)
    for r in results:
        epn = r['graph_edges'] / r['graph_nodes'] if r['graph_nodes'] > 0 else 0
        print(f"  {r['genome_size']:>8,} | {r['graph_nodes']:>8,} | {r['graph_edges']:>8,} | "
              f"{r['entities']:>8,} | {epn:>10.1f}")

    # Maintenance
    print("\n  MAINTENANCE (consolidation + forgetting)")
    print("  " + "-" * 55)
    print(f"  {'Genome':>8} | {'Consol(ms)':>10} | {'Merged':>7} | {'Forget(ms)':>10} | {'Pruned':>7}")
    print("  " + "-" * 55)
    for r in results:
        print(f"  {r['genome_size']:>8,} | {r['consolidate_ms']:>10.1f} | "
              f"{r['consolidated']:>7} | {r['forget_ms']:>10.1f} | {r['forgotten']:>7}")

    # Summary
    print("\n  " + "=" * 60)
    largest = results[-1]
    print(f"  At {largest['genome_size']:,} strands:")
    print(f"    Store:  {largest['store_avg_ms']:.3f} ms/strand  ({largest['store_throughput']:.0f} strands/sec)")
    print(f"    Query:  {largest['query_avg_ms']:.3f} ms avg  (local retrieval)")
    print(f"    Tokens: {largest['avg_tokens_used']:.0f} / {largest['tokens_naive']:,} = {largest['compression_ratio']:.1f}x compression")
    print("  " + "=" * 60)
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cognitive memory scaling benchmark")
    parser.add_argument("--full", action="store_true", help="Run full benchmark (up to 10k)")
    parser.add_argument("--sizes", type=str, default=None,
                        help="Comma-separated genome sizes (e.g. 10,100,1000)")
    args = parser.parse_args()

    if args.sizes:
        sizes = [int(s.strip()) for s in args.sizes.split(",")]
    elif args.full:
        sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    else:
        sizes = [10, 50, 100, 500, 1000]

    results = []
    for size in sizes:
        label = f"{size} strands"
        print(f"  Running benchmark: {label}...", end="", flush=True)
        with Timer() as t:
            r = run_benchmark(size, n_queries=min(20, max(5, size // 10)))
        r["label"] = label
        results.append(r)
        print(f" done ({t.elapsed:.1f}s)")

    print_results(results)

    # Return results for programmatic use
    return results


if __name__ == "__main__":
    main()
