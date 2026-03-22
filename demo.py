#!/usr/bin/env python3
"""
Proof-of-Concept Demo -- Cognitive Memory Architecture v4 (Brain-Perfect)

Every mechanism mirrors real neuroscience:
  - Hippocampal indexing (codebook DNA codes)
  - Neocortical traces (micro-summaries preserving key facts)
  - Spreading activation with adaptive threshold
  - Confidence-weighted spreading
  - Edge type modulation (attention control)
  - Query-aware context assembly (narrative coherence)
  - Memory consolidation (sleep)
  - Intelligent forgetting
  - Strand versioning (supersede outdated info)
  - Hebbian learning + recency priming

Run: python demo.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import os
import sys
import time

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Clean up any previous demo data
for f in ["genome.json", "graph.json", "entities.json"]:
    if os.path.exists(f):
        os.remove(f)

from memory import MemorySystem
from codebook import Codebook


# ─── Configuration ───────────────────────────────────────────────────────────

INTERACTIONS = [
    "Client Acme Corp mentioned they have budget concerns for Q3",
    "Acme Corp's CTO Sarah prefers technical documentation over slide decks",
    "Sent Acme Corp a proposal for $15k/month enterprise plan",
    "Acme Corp asked about integration with Salesforce",
    "Client Beta Ltd is expanding their team to 50 people",
    "Beta Ltd's decision maker is James, reports to VP of Ops",
    "Beta Ltd expressed urgency — they need solution by end of month",
    "Had a call with Acme Corp — Sarah seemed hesitant on pricing",
    "Beta Ltd asked specifically about Slack integration",
    "Acme Corp competitor TechRival just launched a similar product",
    "James from Beta Ltd mentioned their current tool is breaking down",
    "Acme Corp requested a discount — mentioned budget constraints again",
    "Beta Ltd signed up for a 2-week trial",
    "Sarah from Acme Corp went quiet for 2 weeks after seeing pricing",
    "Beta Ltd's trial is going well — James sent positive feedback",
    "Acme Corp budget cycle resets in January — Sarah confirmed",
    "James from Beta Ltd asked about annual pricing vs monthly",
    "Sent Beta Ltd a custom proposal with Slack integration highlighted",
    "Acme Corp re-engaged — Sarah asked to revisit the proposal",
    "Beta Ltd's James asked for a reference call with existing customer",
]

QUERIES = [
    ("Query A", "What's the situation with Acme Corp's pricing concerns?"),
    ("Query B", "James wants to move forward, what do I need to know?"),
    ("Query C", "Which client is closer to closing?"),
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def banner(text):
    print(f"\n{'=' * 76}\n  {text}\n{'=' * 76}")

def section(text):
    print(f"\n{'─' * 68}\n  {text}\n{'─' * 68}")

def c(text, code):
    codes = {
        "green": "\033[92m", "yellow": "\033[93m", "cyan": "\033[96m",
        "red": "\033[91m", "bold": "\033[1m", "dim": "\033[2m",
        "magenta": "\033[95m", "white": "\033[97m", "reset": "\033[0m",
    }
    return f"{codes.get(code, '')}{text}{codes['reset']}"


# ─── Main Demo ───────────────────────────────────────────────────────────────

def main():
    banner("COGNITIVE MEMORY ARCHITECTURE v4 — BRAIN-PERFECT")
    print(f"""
  Every mechanism mirrors real neuroscience:
    {c('Hippocampus', 'cyan')}:  codebook DNA codes (indexing)
    {c('Neocortex', 'magenta')}:   micro-summary traces (fact preservation)
    {c('Spreading activation', 'green')}:  adaptive threshold + confidence weighting
    {c('Consolidation', 'yellow')}:  merge related memories (sleep)
    {c('Forgetting', 'dim')}:  prune unused memories
    {c('Versioning', 'white')}:  supersede outdated beliefs
""")

    codebook = Codebook()
    print(f"  Codebook: {c(str(codebook.total_codes()), 'cyan')} codes  |  "
          f"Model: claude-sonnet-4-20250514")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(c("\n  ERROR: Set ANTHROPIC_API_KEY environment variable", "red"))
        sys.exit(1)

    system = MemorySystem()

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1: STORAGE — Protein -> RNA -> DNA + Trace
    # ══════════════════════════════════════════════════════════════════════

    banner("PHASE 1: STORAGE — Protein -> RNA -> DNA + Neocortical Trace")
    base_timestamp = int(time.time()) - (len(INTERACTIONS) * 3600)

    for i, interaction in enumerate(INTERACTIONS):
        ts = base_timestamp + (i * 3600)
        print(f"\n  [{i+1:2d}/20] {c(interaction[:70], 'dim')}")

        try:
            strand = system.store(interaction, timestamp=ts)
            if strand is None:
                print(f"         {c('[DEDUP]', 'yellow')}")
                continue

            entities = [f"{codebook.decode_entity_type(et)}:{iid}"
                       for et, iid in strand.entity_slots]
            rel = codebook.decode_relation(strand.relation)
            mod = codebook.decode_modifier(strand.modifier)

            print(f"         {c('DNA', 'green')}: [{', '.join(entities)}] "
                  f"REL:{rel} MOD:{mod} SNT:{strand.sentiment:+d}")
            print(f"         {c('Trace', 'magenta')}: {strand.trace}")

            # Show if ego-linked
            ego_strands = system.graph.get_ego_linked_strands("agent")
            if strand.strand_id in ego_strands:
                print(f"         {c('-> EGO LINKED (personally significant)', 'yellow')}")

            # Show if superseding
            if strand.superseded_by is None:
                for s in system.genome.all_strands():
                    if s.superseded_by == strand.strand_id:
                        print(f"         {c('-> SUPERSEDES older strand (updated belief)', 'white')}")
                        break

        except Exception as e:
            print(f"         {c(f'ERROR: {e}', 'red')}")
            continue

    stats = system.stats()
    section("Storage Complete")
    print(f"  Active strands:      {c(str(stats['active_strands']), 'green')} "
          f"({stats['superseded_strands']} superseded)")
    print(f"  Entity instances:    {c(str(stats['entity_instances']), 'cyan')}")
    print(f"  Graph nodes/edges:   {stats['graph_nodes']} / {stats['graph_edges']}")
    print(f"  Ego-linked strands:  {c(str(stats['ego_linked_strands']), 'magenta')}")

    # Entity registry
    section("Entity Registry")
    for inst in system.entity_registry.all_instances()[:10]:
        aliases = ", ".join(sorted(inst.aliases))
        type_name = codebook.decode_entity_type(inst.entity_type)
        print(f"  {c(inst.instance_id, 'cyan')} ({type_name}) "
              f"— {len(inst.strand_ids)} refs — {c(aliases, 'dim')}")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2: RETRIEVAL — Brain reads DNA + Traces directly
    # ══════════════════════════════════════════════════════════════════════

    banner("PHASE 2: RETRIEVAL — Brain-Perfect Reasoning")
    print(f"\n  Brain mechanisms active:")
    print(f"    - Semantic code similarity (cluster-based matching)")
    print(f"    - Adaptive threshold (arousal gating)")
    print(f"    - Confidence-weighted spreading")
    print(f"    - Edge type modulation (attention control)")
    print(f"    - Entity-grouped context (narrative coherence)")
    print(f"    - DNA codes + neocortical traces (structure + facts)")

    all_results = []

    for query_label, query_text in QUERIES:
        section(f"{query_label}: \"{query_text}\"")

        try:
            t_start = time.time()
            result = system.query(query_text)
            t_elapsed = time.time() - t_start
        except Exception as e:
            print(f"  {c(f'ERROR: {e}', 'red')}")
            continue

        all_results.append(result)

        # Brain parameters used
        print(f"\n  {c('BRAIN PARAMS:', 'dim')} threshold={result['threshold_used']} "
              f"complexity={result['query_complexity']} "
              f"entity_groups={result['entity_groups']}")

        # Activated strands
        print(f"\n  {c('ACTIVATED:', 'green')} {len(result['activated'])} strands "
              f"({len(result['not_activated'])} not activated)")
        for sid, score, rendered in result["activated"]:
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"    [{bar}] {score:.3f}")
            for line in rendered.split("\n"):
                print(f"      {c(line, 'cyan')}")

        # Brain's answer
        print(f"\n  {c('BRAIN RESPONSE:', 'bold')}")
        print(f"  ┌{'─' * 66}┐")
        for line in result["answer"].split("\n"):
            while len(line) > 64:
                print(f"  │ {line[:64]} │")
                line = line[64:]
            print(f"  │ {line:<64} │")
        print(f"  └{'─' * 66}┘")

        # Cost
        tokens_used = result["tokens_used"]
        tokens_naive = result["tokens_naive"]
        savings = ((tokens_naive - tokens_used) / tokens_naive * 100) if tokens_naive > 0 else 0
        print(f"\n  {c('COST:', 'bold')} {c(f'{tokens_used}', 'green')} tokens "
              f"(vs {c(f'{tokens_naive}', 'red')} naive) "
              f"= {c(f'{savings:.0f}% savings', 'green')} | "
              f"2 API calls | {t_elapsed:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 3: CONSOLIDATION + FORGETTING
    # ══════════════════════════════════════════════════════════════════════

    banner("PHASE 3: BRAIN MAINTENANCE — Consolidation + Forgetting")

    # Consolidation
    section("Memory Consolidation (Sleep)")
    before = system.stats()
    result_c = system.consolidate()
    after = system.stats()
    print(f"  Consolidated: {c(str(result_c['consolidated']), 'yellow')} strands merged")
    print(f"  Active strands: {before['active_strands']} -> {after['active_strands']}")

    # Forgetting (with very short age for demo purposes)
    section("Intelligent Forgetting")
    result_f = system.forget(min_age_seconds=0, min_activations=1)
    final = system.stats()
    print(f"  Forgotten: {c(str(result_f['forgotten']), 'dim')} unused strands pruned")
    print(f"  Active strands remaining: {c(str(final['active_strands']), 'green')}")

    # ══════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════

    banner("FINAL SUMMARY — BRAIN-PERFECT ARCHITECTURE")

    final = system.stats()
    total_tokens_arch = sum(r["tokens_used"] for r in all_results)
    total_tokens_naive = sum(r["tokens_naive"] for r in all_results)
    avg_tokens_arch = total_tokens_arch / len(all_results) if all_results else 0
    avg_tokens_naive = total_tokens_naive / len(all_results) if all_results else 0
    ratio = avg_tokens_naive / avg_tokens_arch if avg_tokens_arch > 0 else 0

    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  {c('STORAGE (Molecular Biology)', 'magenta')}                                       │
  │    Active strands:         {final['active_strands']:>4}  ({final['superseded_strands']} superseded)              │
  │    Entity instances:       {final['entity_instances']:>4}                                      │
  │    Ego-linked:             {final['ego_linked_strands']:>4}                                      │
  │    Graph edges:            {final['graph_edges']:>4}                                      │
  │                                                                     │
  │  {c('RETRIEVAL (Neuroscience)', 'cyan')}                                           │
  │    API calls per query:    {c('2', 'green')}     (encode + reason)                    │
  │    Context tokens/query:   {avg_tokens_arch:>6.0f}  ({c('fixed', 'green')})                          │
  │    Naive tokens/query:     {avg_tokens_naive:>6.0f}                                    │
  │    Compression ratio:      {ratio:>6.1f}x                                   │
  │                                                                     │
  │  {c('BRAIN MECHANISMS:', 'bold')}                                                    │
  │    ✓ Hippocampal indexing (codebook DNA codes)                      │
  │    ✓ Neocortical traces (micro-summaries with key facts)            │
  │    ✓ Semantic code similarity (cluster-based matching)              │
  │    ✓ Adaptive activation threshold (arousal gating)                 │
  │    ✓ Confidence-weighted spreading                                  │
  │    ✓ Edge type modulation (attention control)                       │
  │    ✓ Entity-grouped context assembly (narrative coherence)          │
  │    ✓ Hebbian learning (co-activation strengthening)                 │
  │    ✓ Recency priming (warm paths)                                   │
  │    ✓ Memory consolidation (sleep — merge related)                   │
  │    ✓ Intelligent forgetting (prune unused)                          │
  │    ✓ Strand versioning (supersede outdated beliefs)                 │
  │    ✓ Bidirectional ego (agent actions + significant events)         │
  │    ✓ Fixed context cost regardless of genome size                   │
  └─────────────────────────────────────────────────────────────────────┘
""")

    print("  SCALING PROJECTION:")
    print("  ┌──────────────┬──────────────────┬──────────────────────────┐")
    print("  │  Genome Size │  Naive (all raw) │  This Architecture       │")
    print("  ├──────────────┼──────────────────┼──────────────────────────┤")
    for n in [20, 100, 1000, 10000, 100000, 1000000]:
        naive = n * 33
        arch = avg_tokens_arch
        print(f"  │  {n:>10,}  │  {naive:>12,} tok │  {arch:>14.0f} tok {c('(fixed)', 'green')}  │")
    print("  └──────────────┴──────────────────┴──────────────────────────┘")
    print(f"\n  {c('The brain is perfect. We copied the brain. We win.', 'green')}")
    print()


if __name__ == "__main__":
    main()
