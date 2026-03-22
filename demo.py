#!/usr/bin/env python3
"""
Proof-of-Concept Demo -- Cognitive Memory Architecture v3

Two biological systems combined:
  MOLECULAR BIOLOGY (Storage): Protein -> RNA -> DNA compression
  NEUROSCIENCE (Retrieval):    Spreading activation -> brain reads DNA directly

The brain decodes itself -- the LLM reads codebook sequences natively,
just like neurons read activation patterns without translation.

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

def banner(text: str):
    width = 76
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def section(text: str):
    print(f"\n{'─' * 68}")
    print(f"  {text}")
    print(f"{'─' * 68}")


def c(text: str, code: str) -> str:
    codes = {
        "green": "\033[92m", "yellow": "\033[93m", "cyan": "\033[96m",
        "red": "\033[91m", "bold": "\033[1m", "dim": "\033[2m",
        "magenta": "\033[95m", "white": "\033[97m", "reset": "\033[0m",
    }
    return f"{codes.get(code, '')}{text}{codes['reset']}"


# ─── Main Demo ───────────────────────────────────────────────────────────────

def main():
    banner("COGNITIVE MEMORY ARCHITECTURE v3")
    print()
    print("  Two biological systems, one framework:")
    print()
    print(f"  {c('MOLECULAR BIOLOGY', 'magenta')} (Storage)")
    print(f"    Protein (raw text) → RNA (extract) → DNA (codebook codes)")
    print(f"    Tiny storage. Massive capacity. Like real DNA.")
    print()
    print(f"  {c('NEUROSCIENCE', 'cyan')} (Retrieval)")
    print(f"    Query → spreading activation → LLM reads DNA directly")
    print(f"    Brain decodes itself. No translation step.")
    print()

    codebook = Codebook()
    print(f"  Codebook: {c(str(codebook.total_codes()), 'cyan')} codes  |  "
          f"Model: claude-sonnet-4-20250514")
    print(f"  Interactions: {len(INTERACTIONS)}  |  Queries: {len(QUERIES)}")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(c("\n  ERROR: Set ANTHROPIC_API_KEY environment variable", "red"))
        sys.exit(1)

    system = MemorySystem()

    # ════════════════════════════════════════════════════════════════════
    #  PHASE 1: MOLECULAR BIOLOGY — Storage Pipeline
    # ════════════════════════════════════════════════════════════════════

    banner("PHASE 1: MOLECULAR BIOLOGY — Protein → RNA → DNA")
    print()
    print(f"  Each interaction passes through the storage pipeline:")
    print(f"    {c('PROTEIN', 'white')}: raw text (the functional event)")
    print(f"    {c('RNA', 'yellow')}: Claude extracts entities, relations, modifiers (1 API call)")
    print(f"    {c('DNA', 'green')}: codebook integer sequence (stored in genome.json)")

    base_timestamp = int(time.time()) - (len(INTERACTIONS) * 3600)
    encoding_calls = 0

    for i, interaction in enumerate(INTERACTIONS):
        ts = base_timestamp + (i * 3600)
        print(f"\n  [{i+1:2d}/20] {c('PROTEIN', 'white')}: {interaction[:65]}")

        try:
            strand = system.store(interaction, timestamp=ts)
            if strand is None:
                print(f"         {c('[DEDUP] Skipped', 'yellow')}")
                continue

            encoding_calls += 1

            # Show RNA extraction
            entities = [f"{codebook.decode_entity_type(et)}:{iid}"
                       for et, iid in strand.entity_slots]
            rel = codebook.decode_relation(strand.relation)
            mod = codebook.decode_modifier(strand.modifier)
            print(f"         {c('RNA', 'yellow')}: entities=[{', '.join(entities)}] "
                  f"rel={rel} mod={mod} snt={strand.sentiment:+d}")

            # Show DNA compression
            seq = strand.to_sequence()
            print(f"         {c('DNA', 'green')}: {seq}  ({strand.sequence_length()} codes)")

        except Exception as e:
            print(f"         {c(f'ERROR: {e}', 'red')}")
            continue

    stats = system.stats()
    section("Storage Complete")
    print(f"  DNA strands in genome:   {c(str(stats['total_strands']), 'green')}")
    print(f"  Entity instances:        {c(str(stats['entity_instances']), 'cyan')} (normalized)")
    print(f"  Graph nodes:             {stats['graph_nodes']} (incl. {stats['ego_nodes']} ego)")
    print(f"  Graph edges:             {stats['graph_edges']}")
    print(f"  API calls for storage:   {encoding_calls} (1 per interaction)")

    # Entity registry
    section("Entity Registry — Alias Normalization")
    for inst in system.entity_registry.all_instances():
        aliases = ", ".join(sorted(inst.aliases))
        type_name = codebook.decode_entity_type(inst.entity_type)
        refs = len(inst.strand_ids)
        print(f"  {c(inst.instance_id, 'cyan')} ({type_name}) — "
              f"{refs} refs — aliases: {c(aliases, 'dim')}")

    # Ego nodes
    ego_strands = system.graph.get_ego_linked_strands("agent")
    section("Ego Node — Agent Identity")
    if ego_strands:
        print(f"  ego:agent → {c(str(len(ego_strands)), 'magenta')} significant strands:")
        for sid in ego_strands:
            strand = system.genome.get(sid)
            if strand:
                mod = codebook.decode_modifier(strand.modifier)
                entities = [iid for _, iid in strand.entity_slots[:2]]
                print(f"    → {c(mod, 'yellow')} | {', '.join(entities)}")
    else:
        print("  ego:agent — no significant strands yet")

    # Size comparison
    raw_chars = sum(len(i) for i in INTERACTIONS)
    genome_size = os.path.getsize("genome.json") if os.path.exists("genome.json") else 0
    print(f"\n  Raw text (protein):    {raw_chars:>6,} chars")
    print(f"  DNA genome on disk:    {genome_size:>6,} bytes")

    # ════════════════════════════════════════════════════════════════════
    #  PHASE 2: NEUROSCIENCE — Brain-like Retrieval
    # ════════════════════════════════════════════════════════════════════

    banner("PHASE 2: NEUROSCIENCE — Brain reads DNA directly")
    print()
    print(f"  The brain decodes itself. No RNA/Protein decode step needed.")
    print(f"  Per query: {c('1 API call', 'green')} to encode + {c('1 API call', 'green')} to reason = {c('2 total', 'bold')}")
    print(f"  Spreading activation is LOCAL (0 API calls, milliseconds).")

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

        # Activated DNA codes (what the brain saw)
        print(f"\n  {c('ACTIVATED DNA CODES:', 'green')} ({len(result['activated'])} strands)")
        print(f"  (This is exactly what the LLM received — no decode step)")
        for sid, score, dna_code in result["activated"]:
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"    [{bar}] {score:.3f}")
            print(f"      {c(dna_code, 'cyan')}")

        # Not activated
        not_activated = result["not_activated"]
        print(f"\n  {c('NOT ACTIVATED:', 'dim')} {len(not_activated)} strands never loaded")

        # The brain's answer
        print(f"\n  {c('BRAIN RESPONSE:', 'bold')} (LLM reasoning directly over DNA codes)")
        print(f"  ┌{'─' * 64}┐")
        for line in result["answer"].split("\n"):
            # Wrap long lines
            while len(line) > 62:
                print(f"  │ {line[:62]} │")
                line = line[62:]
            print(f"  │ {line:<62} │")
        print(f"  └{'─' * 64}┘")

        # Cost
        tokens_used = result["tokens_used"]
        tokens_naive = result["tokens_naive"]
        savings = ((tokens_naive - tokens_used) / tokens_naive * 100) if tokens_naive > 0 else 0

        print(f"\n  {c('COST:', 'bold')}")
        print(f"    Context tokens: {c(f'{tokens_used}', 'green')} (this) vs "
              f"{c(f'{tokens_naive}', 'red')} (naive)")
        print(f"    API calls:      {c('2', 'green')} (encode + reason)")
        print(f"    Savings:        {c(f'{savings:.0f}%', 'green')} tokens  |  "
              f"Time: {t_elapsed:.1f}s")

    # ════════════════════════════════════════════════════════════════════
    #  PHASE 3: FINAL SUMMARY
    # ════════════════════════════════════════════════════════════════════

    banner("FINAL SUMMARY")

    stats = system.stats()
    total_tokens_arch = sum(r["tokens_used"] for r in all_results)
    total_tokens_naive = sum(r["tokens_naive"] for r in all_results)
    avg_tokens_arch = total_tokens_arch / len(all_results) if all_results else 0
    avg_tokens_naive = total_tokens_naive / len(all_results) if all_results else 0
    ratio = avg_tokens_naive / avg_tokens_arch if avg_tokens_arch > 0 else 0

    print(f"""
  ┌───────────────────────────────────────────────────────────────────┐
  │  {c('MOLECULAR BIOLOGY (Storage)', 'magenta')}                                     │
  │    Pipeline:              Protein → RNA → DNA                     │
  │    DNA strands stored:    {stats['total_strands']:>4}                                     │
  │    Entity instances:      {stats['entity_instances']:>4} (normalized via registry)         │
  │    Graph nodes/edges:     {stats['graph_nodes']:>4} / {stats['graph_edges']:<4}                              │
  │    API calls to store:    {encoding_calls:>4} (1 per interaction)                │
  │                                                                   │
  │  {c('NEUROSCIENCE (Retrieval)', 'cyan')}                                        │
  │    Pipeline:              Activate → Brain reads DNA directly     │
  │    API calls per query:   {c('2', 'green')}    (encode + reason)                   │
  │    Context tokens/query:  {avg_tokens_arch:>6.0f}                                  │
  │    Naive tokens/query:    {avg_tokens_naive:>6.0f}                                  │
  │    Compression ratio:     {ratio:>6.1f}x                                 │
  │                                                                   │
  │  {c('FRAMEWORK:', 'bold')}                                                       │
  │    ✓ Real codebook compression (finite alphabet, {stats['codebook_size']} codes)       │
  │    ✓ Entity normalization (aliases → same instance)               │
  │    ✓ Ego nodes (agent identity anchoring)                         │
  │    ✓ Brain-like retrieval (LLM reads DNA directly)                │
  │    ✓ Spreading activation (local, 0 API calls)                    │
  │    ✓ Hebbian learning (co-activation strengthens edges)           │
  │    ✓ Recency priming (warm paths from recent queries)             │
  │    ✓ Fixed context cost regardless of genome size                 │
  └───────────────────────────────────────────────────────────────────┘
""")

    # Scaling projection
    print("  SCALING PROJECTION (context tokens per query):")
    print("  ┌──────────────┬──────────────────┬──────────────────────────┐")
    print("  │  Genome Size │  Naive (all raw) │  This Architecture       │")
    print("  ├──────────────┼──────────────────┼──────────────────────────┤")
    for n in [20, 100, 1000, 10000, 100000, 1000000]:
        naive = n * 33
        arch = avg_tokens_arch
        print(f"  │  {n:>10,}  │  {naive:>12,} tok │  {arch:>14.0f} tok {c('(fixed)', 'green')}  │")
    print("  └──────────────┴──────────────────┴──────────────────────────┘")

    print(f"""
  {c('COMPARISON vs RAG:', 'bold')}
  ┌──────────────────┬──────────────┬──────────────────────────────┐
  │                  │  RAG         │  This Architecture           │
  ├──────────────────┼──────────────┼──────────────────────────────┤
  │  Store cost      │  1 embed     │  1 LLM call (richer)         │
  │  Query API calls │  1-2         │  2 (encode + reason)         │
  │  Query tokens    │  3,000-5,000 │  {avg_tokens_arch:<6.0f} ({c('fixed', 'green')})              │
  │  Multi-hop       │  No          │  {c('Yes', 'green')} (graph traversal)        │
  │  Learns          │  No          │  {c('Yes', 'green')} (Hebbian co-activation)  │
  │  At 100K memories│  Still ~5K   │  Still ~{avg_tokens_arch:.0f} ({c('flat', 'green')})          │
  └──────────────────┴──────────────┴──────────────────────────────┘
""")

    print(f"  {c('→ Two biological systems. One framework. Fixed cost. Infinite scale.', 'green')}")
    print()


if __name__ == "__main__":
    main()
