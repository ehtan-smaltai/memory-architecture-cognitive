# Cognitive Memory Architecture

A biologically-inspired agent memory system with **real DNA-like codebook encoding**, **graph-based spreading activation**, **entity normalization**, and **multi-resolution decoding**.

## The Problem

Current LLM memory systems face a fundamental tradeoff: store more memories → use more context tokens. This architecture breaks that constraint by encoding memories as fixed-width codebook sequences that stay opaque until spreading activation selectively decodes them.

## What Makes This Different

| Property | RAG / MemGPT / SYNAPSE | This Architecture |
|----------|----------------------|-------------------|
| Storage format | Raw text / embeddings / KG triples | **Fixed-width codebook sequences** |
| Compression | None or learned (continuous) | **Discrete finite alphabet (~200 codes)** |
| Retrieval | Similarity search / function calls | **Spreading activation through graph** |
| Decoding | Always full text | **DNA → RNA → Protein (3 resolution levels)** |
| Entity handling | Raw string matching | **Normalized entity registry with aliases** |
| Agent identity | None | **Ego nodes with identity anchoring** |

## Architecture

```
                        ┌─────────────────────────────────┐
                        │         CODEBOOK (v2)           │
                        │  ~200 codes: EntityType,        │
                        │  RelationType, Modifier,        │
                        │  TemporalMarker, Domain         │
                        └──────────┬──────────────────────┘
                                   │
Raw Text ─→ [DNA Encoder] ────────→ CodebookStrand (fixed-width integer sequence)
                │                          │
                │                          ▼
                │              [Entity Registry]
                │              normalize mentions
                │              "Sarah" = "Acme's CTO"
                │                          │
                │                          ▼
                │              [Association Graph]
                │              temporal + entity + semantic
                │              + causal (Hebbian) + ego edges
                │                          │
                │                          ▼
Query ─→ [Expression Engine] ──→ Spreading Activation
                                       │
                            ┌──────────┼──────────┐
                            ▼          ▼          ▼
                         [DNA]      [RNA]    [PROTEIN]
                        0 calls   1 cheap    1 full
                        ~10 tok   ~35 tok    ~75 tok
```

## The Three Layers

### Layer 1: DNA Encoder (`genome.py` + `codebook.py`)
Compresses raw text into **CodebookStrand** units using a finite alphabet of ~200 semantic primitives. The Claude API is constrained to select from codebook codes — no free-form extraction.

A strand is a fixed-width integer sequence:
```
[ORG:1, PERSON:0] → [REL:PRICE_CONCERN:126] → [MOD:NEGATIVE:202] → [TMP:FUTURE:302] → [DOM:SALES:400] → [SNT:500] → [CNF:604]
```

### Layer 2: Association Graph (`graph.py` + `entities.py`)
A weighted directed graph with 5 edge types:
- **temporal** — strands close in time
- **entity_shared** — strands sharing a normalized entity instance
- **semantic** — strands with matching domain + relation codes
- **causal** — co-activated strands (Hebbian: "neurons that fire together wire together")
- **ego** — links from agent identity nodes to personally significant strands

Features:
- **Entity Registry**: normalizes "Sarah", "Acme's CTO", "Sarah Chen" → same instance_id
- **Ego Nodes**: anchor agent identity, auto-link urgent/deadline/escalation strands
- **Recency Priming**: recently activated paths stay "warm" with elevated activation

### Layer 3: Expression Engine (`expression.py`)
Multi-resolution spreading activation retrieval:

1. **SEED** — encode query with codebook, find 3 most similar strands
2. **TRAVERSE** — spread activation through graph (halts when activation < 0.15)
3. **BUDGET** — select top strands within 400-token limit
4. **DECODE** — choose resolution level per strand:
   - **DNA** (activation < 0.4): render codebook codes — **0 API calls, FREE**
   - **RNA** (moderate): 1-sentence structured summary — 1 cheap API call
   - **PROTEIN** (high activation + complex query): full natural language — 1 full API call

## Key Property: Fixed Context Cost

```
Genome Size  │  Naive Cost      │  This Architecture
──────────── │ ──────────────── │ ──────────────────
         20  │         660 tok  │         ~120 tok
        100  │       3,300 tok  │         ~120 tok
      1,000  │      33,000 tok  │         ~120 tok
     10,000  │     330,000 tok  │         ~120 tok
    100,000  │   3,300,000 tok  │         ~120 tok
```

## Cognitive Science Foundations

- **Spreading Activation** — Collins & Loftus (1975): the dominant model of human semantic memory retrieval
- **Hebbian Learning** — "neurons that fire together wire together" (edge strengthening)
- **Encoding Specificity** — Tulving & Thomson (1973): retrieval success depends on encoding context match
- **Complementary Learning Systems** — McClelland et al. (1995): two-tier memory consolidation
- **Miller's Chunking** — compressing memories into minimal, fixed-width units
- **Recursive Language Models** — (arXiv 2512.24601): probe at cheapest level first, escalate when needed

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python demo.py
```

## File Structure

```
memory-architecture-cognitive/
├── codebook.py        # Finite alphabet: ~200 codes across 5 enums
├── entities.py        # Entity instance registry with alias normalization
├── genome.py          # Layer 1: DNA encoder + CodebookStrand persistence
├── graph.py           # Layer 2: Association graph + ego nodes + recency
├── expression.py      # Layer 3: Spreading activation + DNA/RNA/Protein decode
├── memory.py          # Unified MemorySystem interface
├── demo.py            # Proof-of-concept demo
├── requirements.txt
└── README.md
```

## References

- Collins, A.M. & Loftus, E.F. (1975). A spreading-activation theory of semantic processing.
- McClelland, J.L., McNaughton, B.L., & O'Reilly, R.C. (1995). Complementary Learning Systems.
- Tulving, E. & Thomson, D.M. (1973). Encoding specificity and retrieval processes.
- Xu et al. (2026). SYNAPSE: Spreading activation for LLM agent memory. arXiv 2501.01872.
- Jiang et al. (2026). MAGMA: Multi-graph agent memory. arXiv 2601.03236.
- Recursive Language Models. arXiv 2512.24601.

## License

MIT
