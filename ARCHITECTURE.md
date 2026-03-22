# Architecture Deep Dive: Cognitive Memory for Agentic Systems

## What This Is

This is a **context engineering and memory management framework** for AI agentic systems. It combines two biological systems to solve the two fundamental problems of agent memory:

- **Storage**: How do you store unlimited memories cheaply? → Molecular biology (Protein → RNA → DNA compression)
- **Retrieval**: How do you retrieve the right memories efficiently? → Neuroscience (spreading activation + direct code reasoning)

## The Core Insight: External Attention

Modern LLMs use softmax attention internally — for every token, they compute relevance to every other token in the context window. This is essentially "spreading activation" at the neural level. It's brilliant at reasoning over what's in front of it.

**But softmax only works within the context window.**

At 100 memories, you can dump everything in the window and let softmax handle it. At 100,000 memories, you can't. You need something to decide WHICH memories enter the window.

**This architecture is an external attention layer** — a hippocampus that feeds the cortex:

```
┌──────────────────────────────────────────────────────────┐
│  GENOME (unlimited memories on disk)                     │
│  Too big for any context window                          │
│                                                          │
│  ┌──────────────────────────────────┐                    │
│  │  EXTERNAL ATTENTION              │                    │
│  │  (Our architecture)              │                    │
│  │                                  │                    │
│  │  Spreading activation through    │                    │
│  │  association graph selects which  │                    │
│  │  memories enter the context      │                    │
│  │  window. O(edges) not O(n^2).    │                    │
│  └──────────────┬───────────────────┘                    │
│                 │ ~400 tokens (selected)                  │
└─────────────────┼────────────────────────────────────────┘
                  v
┌──────────────────────────────────────────────────────────┐
│  LLM CONTEXT WINDOW (Internal Attention / Softmax)       │
│  Handles reasoning over the selected memories            │
│  The brain reads its own DNA codes natively              │
└──────────────────────────────────────────────────────────┘
```

The hippocampus (our graph) decides what to remember. The cortex (LLM softmax) decides what it means. You need both for a complete cognitive system.

## Two Pipelines

### Storage Pipeline (Molecular Biology)

Just like DNA is the most efficient information storage in the known universe (1 gram = 215 petabytes), we compress memories into minimal codebook sequences.

```
Raw interaction ("Sarah from Acme Corp expressed budget concerns")
         |
    [PROTEIN] — the raw event (input, not stored)
         |
    [RNA TRANSCRIPTION] — Claude API extracts structured fields (1 API call)
         |   entities: [{type: ORG, name: "Acme Corp"}, {type: PERSON, name: "Sarah"}]
         |   relation: PRICE_CONCERN
         |   modifier: NEGATIVE
         |   temporal: FUTURE
         |   domain: SALES
         |   sentiment: -1
         |   confidence: 4
         |
    [DNA COMPRESSION] — map to codebook integer sequence
         |   [1, 0, 126, 202, 302, 400, 501, 604]
         |   8 fixed-width codes
         |
    [GENOME] — stored in genome.json (tiny, scales to millions)
```

The codebook has ~96 codes across 5 dimensions:
- EntityType (17): PERSON, ORG, PRODUCT, METRIC, LOCATION, EVENT, ...
- RelationType (41): WANTS, BLOCKS, CONCERNS, PRICE_CONCERN, WENT_QUIET, ...
- Modifier (13): URGENT, POSITIVE, NEGATIVE, DEADLINE, COMPETITIVE, ...
- TemporalMarker (5): PAST, PRESENT, FUTURE, RECURRING, DEADLINE
- Domain (10): SALES, TECHNICAL, OPS, FINANCE, LEGAL, ...

### Retrieval Pipeline (Neuroscience)

The brain doesn't "decompress" memories before thinking about them. When neurons fire, the activation pattern IS the understanding. Similarly, the LLM reads codebook sequences directly.

```
User query ("Which client is closer to closing?")
         |
    [ENCODE] — query -> DNA codes (1 API call)
         |
    [SPREADING ACTIVATION] — traverse association graph (LOCAL, 0 API calls)
         |   Start from seed nodes (most similar strands)
         |   Spread activation: neighbor = current * edge_weight * 0.7
         |   Stop when activation < 0.15
         |   Add recency bonus from warm paths
         |
    [BUDGET] — select top strands within 400-token limit (LOCAL)
         |
    [BRAIN REASONING] — feed DNA codes directly to LLM (1 API call)
         |   The LLM reads codes like:
         |     [0.85] ORG:beta_ltd | PERSON:james | REL:TRIAL_POSITIVE | SNT:+2
         |     [0.72] ORG:acme_corp | PERSON:sarah | REL:PRICE_CONCERN | SNT:-1
         |   And reasons over them natively.
         |
    [ANSWER] — natural language response
```

Total: 2 API calls per query. ~400 context tokens. Fixed regardless of genome size.

## The Association Graph

Five edge types model different kinds of memory connections:

| Edge Type | Weight | How Created | Biology Analogy |
|-----------|--------|-------------|-----------------|
| temporal | 0.6 | Strands close in time | Episodic memory |
| entity_shared | 0.8 | Strands sharing a normalized entity | Semantic memory |
| semantic | 0.5 | Same domain + relation codes | Categorical association |
| causal | 0.15+ | Co-activated during queries (Hebbian) | "Neurons that fire together wire together" |
| ego | 0.9 | Agent identity anchoring | Self-referential memory |

Key features:
- **Hebbian Learning**: When strands are co-activated in response to a query, their edge weights strengthen. The graph learns which memories are related through usage.
- **Recency Priming**: Recently activated strands stay "warm" for a period, making them easier to activate again. Like how you remember things you just thought about.
- **Ego Nodes**: Special nodes representing the agent's identity, connected to personally significant memories (urgent, deadline, escalation events).
- **Entity Registry**: Normalizes entity mentions across strands. "Sarah", "Acme's CTO", "Sarah Chen" all resolve to the same instance_id.

## When This Architecture Is Necessary vs Unnecessary

### UNNECESSARY (just use long context / RAG)
- Agent with <1,000 total memories
- One-shot tasks with no persistent memory
- All context fits in the current conversation
- Tasks where exact text recall matters (legal, compliance)

### NECESSARY
- Agent operating over weeks/months with accumulating memories
- Memory count will exceed 10,000+
- Relationships between memories matter (multi-hop reasoning)
- The agent needs to learn which memories are associated
- Per-query cost must stay bounded regardless of history length
- Cross-domain reasoning required (sales + technical + organizational)

## Cross-Domain Reasoning

The spreading activation graph doesn't respect domain boundaries. This is a feature, not a bug.

Example: Query "James wants to move forward, what do I need to know?"

The graph activates:
```
James (PERSON)
  --entity_shared--> Beta Ltd trial (SALES domain, +2 sentiment)
  --entity_shared--> VP of Ops (ORGANIZATIONAL domain)
  --temporal--> Current tool breaking down (TECHNICAL domain)
  --entity_shared--> Slack integration request (PRODUCT domain)
  --causal--> Urgency/deadline expressed (SALES domain)
```

The agent connects sales signals + technical problems + org structure + product requirements into a unified recommendation. A vector similarity search would only find documents mentioning "James" or "move forward" — it can't chain across domains.

This is how human cognition works. When you think about a client, you don't isolate "only sales thoughts." Your brain activates everything connected — personality, tech stack, org chart, timeline — and reasons across all of it simultaneously.
