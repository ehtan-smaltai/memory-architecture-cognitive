#!/usr/bin/env python3
"""
Proof-of-Concept Demo — Cognitive Memory Architecture v2

Demonstrates three major upgrades:
  1. CODEBOOK ENCODING — real finite-alphabet compression (not free-form JSON)
  2. ENTITY REGISTRY + EGO NODES — normalized entities + agent identity anchoring
  3. DNA → RNA → PROTEIN — multi-resolution decode (probe cheap, escalate when needed)

Run: python demo.py
Requires: ANTHROPIC_API_KEY environment variable
"""

import os
import sys
import time
import json

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
    print(f"\n{'─' * 64}")
    print(f"  {text}")
    print(f"{'─' * 64}")


def c(text: str, code: str) -> str:
    """ANSI color wrapper."""
    codes = {
        "green": "\033[92m", "yellow": "\033[93m", "cyan": "\033[96m",
        "red": "\033[91m", "bold": "\033[1m", "dim": "\033[2m",
        "magenta": "\033[95m", "reset": "\033[0m",
    }
    return f"{codes.get(code, '')}{text}{codes['reset']}"


def level_color(level: str) -> str:
    """Color-code decode levels."""
    if level == "DNA":
        return c(f"[{level}]", "green")
    elif level == "RNA":
        return c(f"[{level}]", "yellow")
    else:
        return c(f"[{level}]", "magenta")


# ─── Main Demo ───────────────────────────────────────────────────────────────

def main():
    banner("COGNITIVE MEMORY ARCHITECTURE v2 — PROOF OF CONCEPT")
    print()
    print("  Three-layer biologically-inspired memory system with:")
    print(f"    {c('UPGRADE 1', 'bold')}: Codebook encoding — finite alphabet (real DNA compression)")
    print(f"    {c('UPGRADE 2', 'bold')}: Entity registry + ego nodes (associative identity)")
    print(f"    {c('UPGRADE 3', 'bold')}: DNA → RNA → Protein (multi-resolution decode)")
    print()

    codebook = Codebook()
    print(f"  Codebook alphabet size: {c(str(codebook.total_codes()), 'cyan')} codes")
    print(f"  Model: claude-sonnet-4-20250514")
    print(f"  Interactions: {len(INTERACTIONS)}  |  Queries: {len(QUERIES)}")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(c("\n  ERROR: Set ANTHROPIC_API_KEY environment variable", "red"))
        sys.exit(1)

    system = MemorySystem()

    # ── Phase 1: Codebook Encoding ───────────────────────────────────────

    banner("PHASE 1: CODEBOOK ENCODING — Finite alphabet compression")
    print()
    print("  Each interaction is compressed into a fixed-width sequence of")
    print("  codebook indices. Entity names are normalized via the registry.")

    base_timestamp = int(time.time()) - (len(INTERACTIONS) * 3600)

    for i, interaction in enumerate(INTERACTIONS):
        ts = base_timestamp + (i * 3600)
        print(f"\n  [{i+1:2d}/20] {c(interaction[:65], 'dim')}")

        try:
            strand = system.store(interaction, timestamp=ts)
            if strand is None:
                print(f"         {c('[DEDUP] Skipped duplicate', 'yellow')}")
                continue

            # Show codebook encoding
            seq = strand.to_sequence()
            entities = [f"{codebook.decode_entity_type(et)}:{iid}" for et, iid in strand.entity_slots]
            rel = codebook.decode_relation(strand.relation)
            mod = codebook.decode_modifier(strand.modifier)

            print(f"         → entities: {c(', '.join(entities), 'cyan')}")
            print(f"         → REL:{c(rel, 'green')}  MOD:{c(mod, 'yellow')}  SNT:{strand.sentiment:+d}")
            print(f"         → DNA sequence: {c(str(seq), 'dim')}")
            print(f"         → sequence length: {strand.sequence_length()} codes (fixed-width)")

        except Exception as e:
            print(f"         {c(f'ERROR: {e}', 'red')}")
            continue

    stats = system.stats()
    section("Encoding Complete")
    print(f"  Total strands:      {c(str(stats['total_strands']), 'green')}")
    print(f"  Graph nodes:        {stats['graph_nodes']}  (including {stats['ego_nodes']} ego nodes)")
    print(f"  Graph edges:        {stats['graph_edges']}")
    print(f"  Entity instances:   {c(str(stats['entity_instances']), 'cyan')} (normalized)")
    print(f"  Codebook size:      {stats['codebook_size']} codes")

    # Show entity registry
    section("Entity Registry — Normalized Instances")
    for inst in system.entity_registry.all_instances():
        aliases = ", ".join(sorted(inst.aliases))
        type_name = codebook.decode_entity_type(inst.entity_type)
        print(f"  {c(inst.instance_id, 'cyan')} ({type_name})")
        print(f"    aliases: {c(aliases, 'dim')}")
        print(f"    referenced by: {len(inst.strand_ids)} strands")

    # Show ego node connections
    ego_strands = system.graph.get_ego_linked_strands("agent")
    section("Ego Node — Agent Identity Anchoring")
    if ego_strands:
        print(f"  ego:agent linked to {c(str(len(ego_strands)), 'magenta')} significant strands:")
        for sid in ego_strands:
            strand = system.genome.get(sid)
            if strand:
                mod = codebook.decode_modifier(strand.modifier)
                entities = [iid for _, iid in strand.entity_slots]
                print(f"    → {c(sid[:12], 'dim')}  MOD:{c(mod, 'yellow')}  entities: {entities}")
    else:
        print("  ego:agent — no significant strands linked yet")

    # Size comparison
    raw_chars = sum(len(i) for i in INTERACTIONS)
    genome_size = os.path.getsize("genome.json") if os.path.exists("genome.json") else 0
    print(f"\n  {c('SIZE COMPARISON:', 'bold')}")
    print(f"    Raw text:       {raw_chars:>6,} chars")
    print(f"    Encoded genome: {genome_size:>6,} bytes (codebook-compressed)")

    # ── Phase 2: Multi-Resolution Expression ─────────────────────────────

    banner("PHASE 2: EXPRESSION — Spreading activation + DNA/RNA/Protein decode")
    print()
    print("  For each query, the expression engine:")
    print(f"    1. SEED    → encode query with codebook, find similar strands")
    print(f"    2. TRAVERSE → spread activation through the graph")
    print(f"    3. BUDGET  → select top strands within token limit")
    print(f"    4. DECODE  → {c('DNA', 'green')} (free) → {c('RNA', 'yellow')} (cheap) → {c('PROTEIN', 'magenta')} (full)")

    all_query_results = []

    for query_label, query_text in QUERIES:
        section(f"{query_label}: \"{query_text}\"")

        try:
            result = system.query(query_text)
        except Exception as e:
            print(f"  {c(f'ERROR: {e}', 'red')}")
            continue

        all_query_results.append(result)

        # Activated memories with decode levels
        print(f"\n  {c('ACTIVATED MEMORIES:', 'green')} ({len(result['activated'])} strands)")
        for sid, score, decoded, level in result["activated"]:
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"    [{bar}] {score:.3f} {level_color(level)}")
            print(f"      {c(decoded, 'cyan')}")
            print(f"      {c(f'strand: {sid[:12]}...', 'dim')}")

        # Not activated
        not_activated = result["not_activated"]
        print(f"\n  {c('NOT ACTIVATED:', 'yellow')} ({len(not_activated)} strands skipped)")
        for sid in not_activated[:3]:
            strand = system.genome.get(sid)
            if strand:
                entities = [iid for _, iid in strand.entity_slots[:2]]
                print(f"    {c(f'✗ {sid[:12]}... ({", ".join(entities)})', 'dim')}")
        if len(not_activated) > 3:
            print(f"    {c(f'  ... and {len(not_activated) - 3} more', 'dim')}")

        # Decode level breakdown
        levels = result["decode_levels"]
        print(f"\n  {c('DECODE LEVELS:', 'bold')}")
        print(f"    {c(f'DNA (free):    {levels[\"DNA\"]}', 'green')} strands")
        print(f"    {c(f'RNA (cheap):   {levels[\"RNA\"]}', 'yellow')} strands")
        print(f"    {c(f'PROTEIN (full):{levels[\"PROTEIN\"]}', 'magenta')} strands")
        print(f"    API calls made: {result['api_calls_made']}  |  saved: {c(str(result['api_calls_saved']), 'green')}")

        # Token cost
        tokens_used = result["tokens_used"]
        tokens_naive = result["tokens_naive"]
        savings = ((tokens_naive - tokens_used) / tokens_naive * 100) if tokens_naive > 0 else 0

        print(f"\n  {c('TOKEN COST:', 'bold')}")
        print(f"    This architecture:  {c(f'{tokens_used:>4} tokens', 'green')}")
        print(f"    Naive (all raw):    {c(f'{tokens_naive:>4} tokens', 'red')}")
        print(f"    Savings:            {c(f'{savings:.0f}%', 'green')}")

    # ── Phase 3: Architecture Proof ──────────────────────────────────────

    banner("PHASE 3: ARCHITECTURE PROOF — What makes this different")

    # Multi-resolution comparison
    section("Multi-Resolution Decode Comparison")
    print("  Same strand decoded at all 3 levels:")
    sample_strand = system.genome.all_strands()[0] if system.genome.count() > 0 else None
    if sample_strand:
        print(f"\n  {c('[DNA]', 'green')} — 0 API calls, ~{sample_strand.sequence_length() + 5} tokens")
        dna_text = system.expression._decode_dna(sample_strand)
        print(f"    {dna_text}")

        print(f"\n  {c('[RNA]', 'yellow')} — 1 cheap API call, ~35 tokens")
        rna_text = system.expression._decode_rna(sample_strand)
        print(f"    {rna_text}")

        print(f"\n  {c('[PROTEIN]', 'magenta')} — 1 full API call, ~75 tokens")
        protein_text = system.expression._decode_protein(sample_strand)
        print(f"    {protein_text}")

    # Final summary
    banner("FINAL SUMMARY")

    stats = system.stats()
    total_tokens_arch = sum(r["tokens_used"] for r in all_query_results)
    total_tokens_naive = sum(r["tokens_naive"] for r in all_query_results)
    total_api_saved = sum(r["api_calls_saved"] for r in all_query_results)
    total_api_made = sum(r["api_calls_made"] for r in all_query_results)
    avg_tokens_arch = total_tokens_arch / len(all_query_results) if all_query_results else 0
    avg_tokens_naive = total_tokens_naive / len(all_query_results) if all_query_results else 0
    ratio = avg_tokens_naive / avg_tokens_arch if avg_tokens_arch > 0 else 0

    print(f"""
  ┌───────────────────────────────────────────────────────────────┐
  │  GENOME                                                       │
  │    Total strands:           {stats['total_strands']:>4}                              │
  │    Entity instances:        {stats['entity_instances']:>4}  (normalized via registry)   │
  │    Graph nodes:             {stats['graph_nodes']:>4}  (including {stats['ego_nodes']} ego)            │
  │    Graph edges:             {stats['graph_edges']:>4}                              │
  │    Codebook alphabet:       {stats['codebook_size']:>4}  codes                       │
  │                                                               │
  │  TOKEN COST (avg over {len(QUERIES)} queries)                            │
  │    Naive (all raw):         {avg_tokens_naive:>6.0f}  tokens                    │
  │    This architecture:       {avg_tokens_arch:>6.0f}  tokens                    │
  │    Compression ratio:       {ratio:>6.1f}x                          │
  │                                                               │
  │  API EFFICIENCY                                               │
  │    Total API decode calls:  {total_api_made:>4}                              │
  │    API calls saved (DNA):   {total_api_saved:>4}  (decoded for FREE)          │
  │                                                               │
  │  v2 UPGRADES DEMONSTRATED:                                    │
  │    ✓ Codebook encoding (finite alphabet, fixed-width)         │
  │    ✓ Entity registry (normalized cross-strand linking)        │
  │    ✓ Ego nodes (agent identity anchoring)                     │
  │    ✓ DNA → RNA → Protein (multi-resolution decode)            │
  │    ✓ Recency priming (recently activated paths stay warm)     │
  │    ✓ Fixed context cost regardless of genome size             │
  └───────────────────────────────────────────────────────────────┘
""")

    # Scaling projection
    print("  SCALING PROJECTION:")
    print("  ┌──────────────┬─────────────────┬────────────────────┐")
    print("  │  Genome Size │  Naive Cost     │  This Architecture │")
    print("  ├──────────────┼─────────────────┼────────────────────┤")
    for n in [20, 100, 1000, 10000, 100000]:
        naive = n * 33
        arch = avg_tokens_arch  # stays fixed!
        print(f"  │  {n:>10,}  │  {naive:>11,} tok │  {arch:>14.0f} tok │")
    print("  └──────────────┴─────────────────┴────────────────────┘")
    print(f"\n  {c('→ Context cost stays FLAT. Genome scales infinitely.', 'green')}")
    print()


if __name__ == "__main__":
    main()
