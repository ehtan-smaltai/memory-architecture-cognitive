# Framework Comparison: Where We Stand

A comprehensive comparison of this Cognitive Memory Architecture against existing memory systems for AI agents and LLM applications.

## Landscape Overview

The AI agent memory space has several established approaches. Each solves a different subset of the memory problem. None combines all three properties this architecture offers: compressed encoding + graph-based activation retrieval + brain-like direct reasoning.

## Detailed Comparison

### 1. RAG (Retrieval Augmented Generation)

**The standard approach.** Chunk documents, embed them, store in vector DB, retrieve top-K by similarity.

| Aspect | RAG | This Architecture |
|--------|-----|-------------------|
| **Storage** | Raw text chunks + embeddings | Codebook integer sequences (DNA) |
| **Retrieval** | Vector similarity (cosine/dot product) | Spreading activation through graph |
| **Context tokens/query** | 3,000-5,000 (depends on K and chunk size) | ~400 (fixed, budget-capped) |
| **Multi-hop reasoning** | No — retrieves isolated chunks | Yes — graph traversal chains associations |
| **Learning** | No — embeddings frozen at indexing | Yes — Hebbian co-activation strengthens edges |
| **Compression** | None — stores full text + vectors | Yes — ~8-10 integers per memory |
| **Cross-domain linking** | Only if chunks mention same keywords | Yes — entity/temporal/causal edges cross domains |
| **Persistence** | Vector DB (Pinecone, Weaviate, etc.) | JSON files (portable, no infra) |
| **Setup complexity** | High (vector DB + embedding model + chunking strategy) | Low (pip install, one JSON file) |
| **Proven at scale** | Yes — battle-tested in production | No — proof of concept |

**RAG wins when**: You have a static document corpus, need exact text recall, want battle-tested reliability, or need sub-second latency.

**We win when**: Memories are relational (people/events connected over time), you need multi-hop reasoning, memory count exceeds context limits, or the agent needs to learn from usage patterns.

---

### 2. MemGPT / Letta

**OS-inspired virtual memory.** Main context (RAM) holds the active working set. Archival storage (disk) holds everything else. The LLM manages its own memory via function calls.

| Aspect | MemGPT/Letta | This Architecture |
|--------|-------------|-------------------|
| **Storage** | Tiered: context (RAM) + DB + vector store | Single tier: codebook DNA in genome.json |
| **Retrieval** | LLM-driven function calls (archival_memory_search, core_memory_append) | Spreading activation (automatic, no function calls) |
| **Context tokens/query** | 2,000-4,000 (main context buffer) | ~400 (fixed) |
| **Compression** | Summaries only (recursive summarization) | Real compression (codebook encoding) |
| **Multi-hop** | Limited — LLM must explicitly chain searches | Automatic — graph traversal chains naturally |
| **Learning** | No edge strengthening | Yes — Hebbian learning |
| **Graph-based** | No | Yes |
| **Autonomy** | LLM decides when to search (can miss) | Automatic activation (deterministic) |

**MemGPT wins when**: You need the agent to actively manage its own memory, want a mature framework with tooling, or need fine-grained control over what's in context.

**We win when**: You want automatic retrieval without relying on the LLM to "remember to search," need multi-hop associations, or want the graph to learn over time.

---

### 3. SYNAPSE (arXiv 2501.01872, January 2026)

**The closest competitor.** Uses spreading activation for LLM agent memory retrieval. Graph nodes are text, edges represent associations.

| Aspect | SYNAPSE | This Architecture |
|--------|---------|-------------------|
| **Storage** | Graph nodes as text + embeddings | Codebook integer sequences (DNA) |
| **Retrieval** | Spreading activation + embeddings | Spreading activation (codebook similarity) |
| **Compressed?** | No — stores full text | Yes — fixed-width codebook codes |
| **Graph-based?** | Yes | Yes |
| **Activation-based?** | Yes | Yes |
| **Entity normalization** | Not mentioned | Yes — entity registry with alias resolution |
| **Ego nodes** | No | Yes — agent identity anchoring |
| **Recency priming** | Not mentioned | Yes — warm paths from recent queries |
| **Brain-like decode** | No — retrieves raw text | Yes — LLM reads codes directly |
| **Context cost** | Varies with retrieved text length | Fixed (budget-capped codebook codes) |

**SYNAPSE wins when**: You want an academically validated approach with benchmark results on LoCoMo and LongMemEval.

**We win when**: You need compressed storage (SYNAPSE stores full text), want fixed context cost (SYNAPSE's cost varies), or need entity normalization and ego nodes. Our key differentiation is the DNA encoding layer — SYNAPSE has no compression.

---

### 4. HippoRAG

**Hippocampus-inspired RAG.** Uses knowledge graph triples + personalized PageRank for retrieval. Named after the hippocampus's role in memory consolidation.

| Aspect | HippoRAG | This Architecture |
|--------|----------|-------------------|
| **Storage** | KG triples + passages | Codebook sequences |
| **Retrieval** | Personalized PageRank | Spreading activation |
| **Compressed?** | No | Yes |
| **Graph-based?** | Yes | Yes |
| **Activation-based?** | Partially (PageRank is related) | Yes (Collins & Loftus spreading activation) |
| **Multi-hop** | Yes (through KG) | Yes (through association graph) |
| **Learning** | No — static KG | Yes — Hebbian learning |
| **Entity handling** | KG entity extraction | Codebook entity registry |

**HippoRAG wins when**: You have structured knowledge that maps well to KG triples, need strong multi-hop over factual relationships.

**We win when**: Memories are experiential (conversations, events, sentiments) rather than factual, or you need the graph to evolve with usage.

---

### 5. GraphRAG (Microsoft, 2024)

**Hierarchical graph-based RAG.** Builds community structures via Leiden clustering, generates pre-computed summaries.

| Aspect | GraphRAG | This Architecture |
|--------|----------|-------------------|
| **Storage** | Temporal KG in Neo4j (~600K+ tokens) | Codebook JSON (~100 bytes per memory) |
| **Retrieval** | Hybrid semantic + BM25 + graph | Spreading activation |
| **Compressed?** | No (600K+ tokens for moderate corpus) | Yes |
| **Batch-oriented?** | Yes — requires full recomputation on updates | No — incremental (add one strand at a time) |
| **Real-time updates** | No — expensive reindexing | Yes — instant (add node + edges) |
| **Infrastructure** | Neo4j + embedding model + clustering | JSON files only |

**GraphRAG wins when**: You have a large static corpus needing global summarization, like document analysis or research.

**We win when**: Memories arrive as a stream (conversations, events), you need real-time updates, or you want zero infrastructure.

---

### 6. Mem0 / Mem0g

**Memory management for AI agents.** Graph-based memory with vector similarity retrieval.

| Aspect | Mem0 | This Architecture |
|--------|------|-------------------|
| **Storage** | Vector + KV + graph | Codebook sequences |
| **Retrieval** | Vector similarity + graph traversal | Spreading activation |
| **Compressed?** | Fact extraction (partial) | Yes — codebook compression |
| **Graph-based?** | Yes | Yes |
| **Activation-based?** | No | Yes |
| **API** | Hosted SaaS | Self-hosted (JSON files) |
| **Cost model** | Usage-based pricing | Your own API costs only |

**Mem0 wins when**: You want a managed service, need production-ready APIs, or don't want to self-host.

**We win when**: You want full control, need codebook compression, or want spreading activation instead of vector similarity.

---

### 7. Generative Agents (Stanford/Google, UIST 2023)

**The reflection-based approach.** Flat memory stream with recency-importance-relevance scoring. Pioneered the "generative agent" concept.

| Aspect | Generative Agents | This Architecture |
|--------|-------------------|-------------------|
| **Storage** | Flat memory stream (natural language) | Codebook sequences in graph |
| **Retrieval** | Tri-score: recency + importance + relevance | Spreading activation + recency priming |
| **Graph structure** | None — flat sequential | Yes — directed weighted graph |
| **Compression** | None | Yes |
| **Multi-hop** | No — flat retrieval | Yes — graph traversal |
| **Reflection** | Yes — observations -> reflections -> meta-reflections | No (could be added) |

**Generative Agents wins when**: You're building social simulations, need the reflection/abstraction mechanism.

**We win when**: You need multi-hop retrieval (impossible with flat storage), compressed storage, or graph-based associations.

---

### 8. LangChain / LlamaIndex Memory

**Framework-level memory modules.** ConversationBufferMemory, ConversationSummaryMemory, VectorStoreRetrieverMemory, etc.

| Aspect | LangChain Memory | This Architecture |
|--------|-----------------|-------------------|
| **Storage** | Various (buffer, summary, vector store) | Codebook sequences |
| **Retrieval** | Configurable (recency, similarity, etc.) | Spreading activation |
| **Compression** | Summary memory (lossy text compression) | Codebook encoding (structured compression) |
| **Graph-based?** | No (unless custom) | Yes |
| **Multi-hop** | No | Yes |
| **Learning** | No | Yes — Hebbian |
| **Ecosystem** | Massive (plugins, integrations, docs) | Standalone |

**LangChain wins when**: You need ecosystem integration, want plug-and-play components, or need mature tooling.

**We win when**: You need a fundamentally different retrieval mechanism (graph-based activation vs. similarity search).

---

## Summary Matrix

| System | Compressed? | Graph? | Activation? | Learns? | Multi-hop? | Fixed Cost? | Brain-like? |
|--------|------------|--------|-------------|---------|-----------|-------------|-------------|
| RAG | No | No | No | No | No | No | No |
| MemGPT/Letta | Summaries | No | No | No | Limited | No | No |
| SYNAPSE | No | Yes | Yes | Hebbian | Yes | No | No |
| HippoRAG | No | Yes | Partial | No | Yes | No | No |
| GraphRAG | No | Yes | No | No | Yes | No | No |
| Mem0 | Partial | Yes | No | No | Limited | No | No |
| Gen. Agents | No | No | No | No | No | No | No |
| LangChain | Summaries | No | No | No | No | No | No |
| **This Architecture** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

## Our Unique Combination

No existing system combines ALL of these:
1. **Compressed encoding** (codebook DNA — not raw text, not embeddings)
2. **Graph-based retrieval** (spreading activation — not vector similarity)
3. **Brain-like reasoning** (LLM reads codes directly — no decode step)
4. **Hebbian learning** (graph evolves with usage)
5. **Fixed context cost** (budget-capped regardless of genome size)
6. **Entity normalization** (alias registry across memories)
7. **Ego nodes** (agent identity anchoring)

Each individual component exists in prior work. The combination is novel.

## Honest Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **No benchmarks** | Can't claim superiority without LoCoMo/LongMemEval results | Need to benchmark against SYNAPSE and HippoRAG |
| **Lossy compression** | Codebook encoding loses nuance (tone, duration, sequence) | Expandable codebook + optional protein-level decode as fallback |
| **Encoding cost** | 1 LLM API call per memory stored (expensive vs. embedding) | Could use smaller/local models for encoding |
| **Latency** | 2 API calls per query (vs RAG's 1) | Graph traversal is local; API calls are parallelizable |
| **Codebook rigidity** | Fixed ~96 codes may not cover all domains | Codebook is extensible; add domain-specific codes |
| **No production hardening** | JSON files, no concurrent access, no auth | This is a proof of concept, not production software |

---

## Benchmarks in the Field

### LoCoMo Benchmark
- 300-turn conversations spanning up to 35 sessions (~9K tokens avg)
- Tests: QA, event summarization, multi-modal dialogue
- Temporal grounding: up to 25 events over 6-12 months
- Best performer: **SYNAPSE** with significant advantages on multi-hop reasoning
- Mem0 score: 67.13% (LLM-as-a-Judge)

### LongMemEval Benchmark
- 500 manually-created questions testing 5 core abilities
- Short variant: ~115K tokens, ~200 dialogue turns
- Medium variant: ~1.5M tokens, ~2000 turns, 500 sessions
- Tests: information extraction, multi-session reasoning, temporal reasoning, knowledge updates

### MemoryAgentBench
- Tests 4 competencies: accurate retrieval, test-time learning, long-range understanding, conflict resolution
- Key finding: **multi-hop conflict resolution shows dramatic failures across all systems (max 6% accuracy)**

### Published Performance Numbers

| System | Token Savings | Latency Improvement | Multi-hop QA |
|--------|-------------|--------------------|--------------|
| Mem0 | 90% (1,764 vs 26,031 tokens) | 91% p95 reduction (0.200s) | Supported |
| HippoRAG | 10-30x cheaper than iterative RAG | 6-13x faster | 20% improvement |
| SYNAPSE | Graph-optimized | Good | Best on LoCoMo |
| Zep | Graph-optimized | 90% reduction | Temporal reasoning |
| LazyGraphRAG | 0.1% cost of full GraphRAG | Good | Global queries |

### Where We Would Need to Prove Ourselves
- Run against LoCoMo (multi-hop QA, temporal reasoning)
- Run against LongMemEval (1.5M token conversations)
- Compare codebook compression ratio vs Mem0's fact extraction
- Benchmark spreading activation vs HippoRAG's Personalized PageRank
- Measure p95 latency for graph traversal + reasoning

---

## Cost Analysis (Per Query)

Assuming Claude Sonnet pricing (~$3/M input, ~$15/M output):

| System | Input tokens | Output tokens | Est. cost per query |
|--------|-------------|--------------|-------------------|
| RAG (top-10 chunks) | ~5,000 | ~500 | ~$0.022 |
| MemGPT (3 function calls) | ~3,000 x 3 | ~200 x 3 | ~$0.036 |
| Mem0 | ~1,764 | ~300 | ~$0.010 |
| This Architecture | ~600 (encode) + ~600 (reason) | ~50 + ~300 | ~$0.009 |

At scale (1,000 queries/day):
- RAG: ~$22/day
- MemGPT: ~$36/day
- Mem0: ~$10/day
- This: ~$9/day

Infrastructure costs:
- RAG: $25-300/month (vector DB hosting — Pinecone, Weaviate, Qdrant)
- GraphRAG: Neo4j instance + compute for Leiden clustering
- Mem0: Open-source or enterprise pricing
- Zep: Platform service pricing
- This Architecture: **$0 infrastructure** (JSON files, no external DB)

The cost advantage grows as memory count increases because our context tokens stay fixed while RAG's grow with chunk retrieval.

---

## Industry Adoption (as of 2026)

| System | Adoption | GitHub Stars | Notable Users |
|--------|----------|-------------|---------------|
| RAG | Ubiquitous | N/A (pattern, not library) | Everyone |
| LangChain | Very High | 100K+ | Enterprise standard |
| LlamaIndex | High | 37K+ | Enterprise standard |
| Mem0 | Growing | 37K+ | Netflix, Lemonade, Rocket Money |
| GraphRAG | Moderate | 20K+ | Microsoft ecosystem |
| MemGPT/Letta | Moderate | 12K+ | Research + startups |
| Zep | Growing | 3K+ | Agentic platforms |
| SYNAPSE | Research | N/A | Academic |
| HippoRAG | Research | N/A | Academic |
| **This Architecture** | **New** | **New** | **Proof of concept** |

---

## Emerging Trends (2025-2026)

Key trends that validate our architectural choices:

1. **Graph-based memory outperforming vectors** — SYNAPSE, HippoRAG, Zep, Mem0 all moving toward graph structures. Vector-only retrieval is being surpassed.

2. **Temporal awareness becoming standard** — Zep's bi-temporal KG, SYNAPSE's temporal decay. Static embeddings can't capture when things happened.

3. **Multi-hop still unsolved** — MemoryAgentBench shows max 6% accuracy on multi-hop conflict resolution. This remains the hardest open problem.

4. **Compression is the next frontier** — Mem0's 90% token savings, Focus framework's 22.7% net savings, sparsification achieving 8x smaller memory. Everyone is trying to compress.

5. **Always-On Memory** — Google exploring ditching vector databases entirely for LLM-driven persistent memory. Validates the brain-like approach.

---

## Positioning

```
                    Simple <------------------------> Complex

  Buffer Memory ---- RAG ---- MemGPT ---- GraphRAG ---- This Architecture
       |              |          |            |                |
    No persistence  Static    Managed      Static          Living graph
    No compression  chunks    tiered       community       Compressed DNA
    No graph        vectors   function     hierarchical    Spreading activation
                              calls        summaries       Brain-like reasoning
                                                          Hebbian learning
```

**We sit at the complex end** — more moving parts, but genuinely novel capabilities that simpler systems can't match. The question is whether your use case needs those capabilities.

---

## When to Use What

| Your Situation | Best Choice |
|---------------|-------------|
| Document Q&A over a knowledge base | **RAG** |
| Chatbot with short conversation history | **LangChain Buffer Memory** |
| Agent that manages its own memory explicitly | **MemGPT/Letta** |
| Large corpus analysis and summarization | **GraphRAG** |
| Academic research on memory retrieval | **SYNAPSE** |
| Managed memory service with API | **Mem0** |
| Temporal-aware agentic memory | **Zep** |
| Long-running agent with relational memories, multi-hop reasoning, fixed cost, and learning | **This Architecture** |

---

## References

- [SYNAPSE: Spreading Activation for LLM Memory](https://arxiv.org/abs/2601.02744) (January 2026)
- [HippoRAG: Neurobiologically Inspired LTM for LLMs](https://arxiv.org/abs/2405.14831)
- [GraphRAG Documentation](https://microsoft.github.io/graphrag/)
- [Mem0: Scalable Long-Term Memory for AI Agents](https://arxiv.org/abs/2504.19413)
- [Zep: Temporal KG Architecture for Agent Memory](https://arxiv.org/abs/2501.13956)
- [Generative Agents: Interactive Simulacra](https://arxiv.org/abs/2304.03442)
- [LoCoMo: Evaluating Long-Term Conversational Memory](https://snap-research.github.io/locomo/)
- [Agentic Memory: Unified LTM/STM Management](https://arxiv.org/abs/2601.01885)
- [Recursive Language Models](https://arxiv.org/html/2512.24601v1)
- [MAGMA: Multi-Graph Agent Memory](https://arxiv.org/abs/2601.03236) (January 2026)
