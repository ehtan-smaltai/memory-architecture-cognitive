# Cognitive Memory Architecture

Two biological systems combined into one framework for AI agent memory.

**Molecular Biology** solves storage — Protein → RNA → DNA compression.
**Neuroscience** solves retrieval — spreading activation + brain reads DNA directly.

## The Core Idea

DNA is the most efficient information storage in the known universe — 1 gram stores 215 petabytes. The brain is the most efficient retrieval system — millisecond recall without "decompressing" anything.

This architecture uses **molecular biology for how memories are stored** and **neuroscience for how memories are retrieved**. DNA sits at the intersection: the output of storage AND the input to retrieval.

## Architecture

```
STORAGE (Molecular Biology)              RETRIEVAL (Neuroscience)
━━━━━━━━━━━━━━━━━━━━━━━━━━              ━━━━━━━━━━━━━━━━━━━━━━━

Raw interaction                          User query
    ↓                                        ↓
[PROTEIN] — the event itself             [ENCODE] — query → DNA codes
    ↓                                        ↓
[RNA] — extract entities,                [ACTIVATE] — spreading activation
        relations, modifiers                 through association graph
        (1 API call)                         (LOCAL — 0 API calls)
    ↓                                        ↓
[DNA] — codebook integer sequence        [REASON] — feed DNA codes directly
        (stored in genome.json)              to LLM (1 API call)
                                             ↓
        ←─── DNA connects both ───→      Brain reads codes natively.
                                         No decode step needed.
```

## Why Brain-Like Retrieval?

The brain doesn't "decompress" memories before thinking about them. When neurons fire, the **activation pattern IS the understanding**. Similarly, the LLM reads codebook sequences directly — no translation back to English needed.

```
OLD approach (v1-v2):
  DNA codes → [API call to decode to English] → feed to LLM → answer
  Cost: 4-6 API calls per query

BRAIN-LIKE approach (v3):
  DNA codes → feed directly to LLM → answer
  Cost: 2 API calls per query (encode + reason)
```

## Cost Comparison

```
                    Store          Query              Context tokens
                    ─────          ─────              ──────────────
RAG:                1 embed        1 LLM call         ~3,000-5,000
MemGPT:             1 LLM call     2-5 function calls ~2,000-4,000
This architecture:  1 LLM call     2 LLM calls        ~200-400 (FIXED)
```

### Scaling

| Genome Size | Naive Cost | RAG | This Architecture |
|------------|-----------|-----|-------------------|
| 20 | 660 tok | ~3,000 tok | ~200 tok |
| 100 | 3,300 tok | ~4,000 tok | ~200 tok |
| 1,000 | 33,000 tok | ~5,000 tok | ~200 tok |
| 10,000 | 330,000 tok | ~5,000 tok | ~200 tok |
| 100,000 | 3.3M tok | ~5,000 tok | ~200 tok |
| 1,000,000 | 33M tok | ~5,000 tok | **~200 tok** |

RAG's token cost grows with chunk size and K. This architecture's cost is **fixed** — determined only by the token budget, not the genome size.

## The Three Storage Layers

### Protein (Input)
The raw interaction — the functional event that happened. This is the input to the storage pipeline. Like biological proteins that do the actual work in cells.

### RNA Transcription (`genome.py`)
Claude API extracts structured fields from raw text: entities, relations, modifiers, sentiment, temporal markers. This is the transcription step — converting functional events into a structured intermediate form. **1 API call per memory.**

### DNA Compression (`codebook.py`)
Maps extracted fields to a finite alphabet of ~96 codebook codes across 5 dimensions:
- **EntityType** (17 codes): PERSON, ORG, PRODUCT, METRIC, ...
- **RelationType** (41 codes): WANTS, BLOCKS, CONCERNS, PRICE_CONCERN, ...
- **Modifier** (13 codes): URGENT, POSITIVE, NEGATIVE, DEADLINE, ...
- **TemporalMarker** (5 codes): PAST, PRESENT, FUTURE, RECURRING, DEADLINE
- **Domain** (10 codes): SALES, TECHNICAL, OPS, FINANCE, ...

A memory becomes a fixed-width integer sequence: `[1, 0, 26, 2, 1, 0, 499, 604]`

### Entity Registry (`entities.py`)
Normalizes entity mentions across strands. "Sarah", "Acme's CTO", "Sarah Chen" all resolve to the same `instance_id` via fuzzy alias matching.

## The Retrieval Layer

### Association Graph (`graph.py`)
A weighted directed graph with 5 edge types:
- **temporal** — strands close in time
- **entity_shared** — strands sharing normalized entity instances
- **semantic** — strands with matching domain + relation codes
- **causal** — co-activated strands (Hebbian learning)
- **ego** — agent identity anchoring to significant strands

Features: ego nodes, recency priming, Hebbian co-activation learning.

### Expression Engine (`expression.py`)
Brain-like retrieval in 4 steps:
1. **SEED** — encode query, find similar strands (local)
2. **ACTIVATE** — spreading activation through graph (local, 0 API calls)
3. **BUDGET** — select top strands within token limit (local)
4. **REASON** — feed DNA codes directly to LLM (1 API call)

The LLM reads codebook sequences natively. No decode step. The brain decodes itself.

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python demo.py
```

## File Structure

```
memory-architecture-cognitive/
├── codebook.py        # DNA alphabet: ~96 codes, CodebookStrand dataclass
├── entities.py        # Entity normalization with alias registry
├── genome.py          # Storage pipeline: Protein → RNA → DNA
├── graph.py           # Association graph + ego nodes + recency priming
├── expression.py      # Brain-like retrieval: activate → reason over DNA
├── memory.py          # Unified MemorySystem (storage + retrieval)
├── demo.py            # Proof-of-concept demo
├── requirements.txt
└── README.md
```

## Cognitive Science Foundations

- **Spreading Activation** — Collins & Loftus (1975): human semantic memory retrieval
- **Hebbian Learning** — "neurons that fire together wire together"
- **Encoding Specificity** — Tulving & Thomson (1973): retrieval depends on encoding match
- **Complementary Learning Systems** — McClelland et al. (1995): two-tier consolidation
- **Miller's Chunking** — compressing to fixed-width minimal units
- **Recursive Language Models** — arXiv 2512.24601: probe cheapest level first

## License

MIT
