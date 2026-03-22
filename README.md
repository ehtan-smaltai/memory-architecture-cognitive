# Cognitive Memory Architecture

A biologically-inspired agent memory system with **DNA-like encoding**, **graph-based spreading activation**, and **activation-selective decoding**.

## The Problem

Current LLM memory systems face a fundamental tradeoff: store more memories → use more context tokens. This architecture breaks that constraint.

## Architecture

Three integrated layers, each solving a distinct problem:

```
Raw Text → [DNA Encoder] → Compressed Strand → [Association Graph] → Connected Genome
                                                        ↓
User Query → [Expression Engine] → Spreading Activation → Selective Decode → Context
```

### Layer 1: DNA Encoder (`genome.py`)
Compresses raw interactions into minimal encoded units ("strands"). Each strand captures entities, relations, sentiment, confidence, and domain — never raw text.

### Layer 2: Association Graph (`graph.py`)
A weighted directed graph over strands implementing:
- **Temporal edges** — strands close in time
- **Entity-shared edges** — strands sharing named entities
- **Semantic edges** — strands in the same domain
- **Causal edges** — co-activated strands (Hebbian learning)

### Layer 3: Expression Engine (`expression.py`)
Spreading activation retrieval that **never loads all strands**:
1. **SEED** — encode query, find closest strands
2. **TRAVERSE** — spread activation through the graph
3. **BUDGET** — select top strands within token limit
4. **DECODE** — convert selected strands to natural language

## Key Property

**Context window token cost stays FIXED regardless of how many memories are stored.**

```
Genome Size  │  Naive Cost     │  This Architecture
──────────── │ ─────────────── │ ──────────────────
         20  │       660 tok   │         ~120 tok
        100  │     3,300 tok   │         ~120 tok
      1,000  │    33,000 tok   │         ~120 tok
     10,000  │   330,000 tok   │         ~120 tok
```

## Cognitive Science Foundations

- **Spreading Activation** — Collins & Loftus (1975): dominant model of human semantic memory retrieval
- **Hebbian Learning** — "neurons that fire together wire together"
- **Encoding Specificity** — Tulving & Thomson (1973): retrieval depends on encoding context
- **Complementary Learning Systems** — McClelland et al. (1995): two-tier memory consolidation
- **Miller's Chunking** — compressing memories into minimal units

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY=your-key-here

# Run the demo
python demo.py
```

## File Structure

```
memory_architecture/
├── genome.py          # Layer 1: DNA encoder + strand management
├── graph.py           # Layer 2: Association graph + Hebbian learning
├── expression.py      # Layer 3: Spreading activation + decode
├── memory.py          # Unified MemorySystem interface
├── demo.py            # Proof-of-concept demo (20 interactions, 3 queries)
├── requirements.txt   # Dependencies
└── README.md
```

## How It Differs from Existing Systems

| System | Storage | Retrieval | Compressed? | Graph? | Activation? |
|--------|---------|-----------|-------------|--------|-------------|
| SYNAPSE | Graph nodes (text) | Spreading activation + embeddings | No | Yes | Yes |
| HippoRAG | KG triples + passages | Personalized PageRank | No | Yes | Partially |
| MemGPT/Letta | Tiered (context + DB) | LLM-driven function calls | Summaries | No | No |
| GraphRAG | Temporal KG (Neo4j) | Hybrid semantic + BM25 | No (600K+) | Yes | No |
| **This Architecture** | **Compressed encoded units** | **Spreading activation** | **Yes (DNA-like)** | **Yes** | **Yes** |

## References

- Collins, A.M. & Loftus, E.F. (1975). A spreading-activation theory of semantic processing.
- McClelland, J.L., McNaughton, B.L., & O'Reilly, R.C. (1995). Complementary Learning Systems.
- Tulving, E. & Thomson, D.M. (1973). Encoding specificity and retrieval processes.
- SYNAPSE (Xu et al., arXiv 2501.01872, January 2026)
- MAGMA (Jiang et al., arXiv 2601.03236, January 2026)

## License

MIT
