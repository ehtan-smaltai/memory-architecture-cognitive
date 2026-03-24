# Technical Specification: Cognitive Memory Architecture

## 1. System Overview

A two-pipeline memory architecture for AI agents combining lossy codebook
compression (storage) with graph-based spreading activation (retrieval).

**Invariants:**
- Storage cost: 1 LLM call per memory
- Retrieval cost: 2 LLM calls per query (encode + reason), fixed context tokens
- All retrieval graph operations are local (zero LLM calls)
- Context tokens per query are bounded by TOKEN_BUDGET (500), independent of genome size

---

## 2. Data Model

### 2.1 CodebookStrand (the atom)

| Field | Type | Range | Purpose |
|---|---|---|---|
| strand_id | UUID4 string | - | Unique identifier |
| entity_slots | list[tuple[int, str]] | 1-4 entries | (EntityType code, instance_id) |
| relation | int | 0-99 | RelationType code |
| modifier | int | 0-15 | Modifier code |
| temporal | int | 0-4 | TemporalMarker code |
| domain | int | 0-9 | Domain code |
| sentiment | int | -2 to +2 | Quantized polarity |
| confidence | int | 1 to 5 | Quantized certainty |
| timestamp | int | Unix epoch | Creation time |
| raw_hash | SHA-256 hex | - | Deduplication key |
| trace | string | 10-15 words | Neocortical micro-summary |
| activation_count | int | ≥0 | Times activated in queries |
| superseded_by | string \| None | - | Strand that replaces this one |

**Size:** 8-12 integer codes + ~15-word trace per strand. Estimated 17-25 tokens
when rendered for LLM reasoning.

### 2.2 Codebook Dimensions

| Dimension | Code Count | Fallback | Purpose |
|---|---|---|---|
| EntityType | 17 | UNKNOWN (99) | What kind of entity |
| RelationType | 41 | OTHER (99) | Relationship between entities |
| Modifier | 13 | OTHER (15) | Contextual urgency/sentiment |
| TemporalMarker | 5 | PRESENT (1) | When the memory relates to |
| Domain | 10 | GENERAL (8) | Coarse topic classification |
| Sentiment | 5 | 0 | -2 to +2 quantized |
| Confidence | 5 | 3 | 1-5 quantized |

**Total vocabulary:** ~96 semantic codes.

### 2.3 Association Graph

Weighted directed graph (networkx.DiGraph). Nodes are strand_ids or ego node IDs.

| Edge Type | Initial Weight | Created By | Semantics |
|---|---|---|---|
| temporal | 0.6 | Auto, 3 most recent strands | Episodic proximity |
| entity_shared | 0.8 × shared_count | Auto, EntityRegistry lookup | Shared entity reference |
| semantic | 0.5 | Auto, domain+relation index | Same domain + relation code |
| causal | 0.15 | Hebbian co-activation | Learned association |
| ego | 0.9 | Auto, modifier/relation rules | Agent identity anchoring |

**Constraints:**
- MAX_EDGES_PER_NODE = 50 (weakest non-ego edges pruned on overflow)
- DECAY_FACTOR = 0.99 per query (with floor — see mitigation R6)
- Edges are bidirectional (two directed edges per logical connection)

### 2.4 Entity Registry

Maps raw entity mentions → canonical instance_ids via:
1. Exact alias match (O(1) dict lookup)
2. Fuzzy substring match (same entity_type, min length 3, length ratio ≥ 0.6)
3. New instance creation

---

## 3. Storage Pipeline

```
store(raw_text)
  ├── SHA-256 dedup check (genome.has_hash)
  ├── BEGIN BATCH (suppress disk writes)
  ├── LLM extraction with retry (max 3 attempts)
  │     ├── Constrained JSON prompt → codebook codes
  │     ├── Parse with fence stripping + JSON extraction
  │     └── Validate required keys (entities, relation)
  │     └── On total failure: fallback to UNKNOWN/OTHER + raw_text[:80] as trace
  ├── Entity resolution (1-4 entities through registry)
  ├── Supersede check (same entity + relation + different sentiment)
  ├── Genome.add(strand)
  ├── Graph.add_strand (temporal + entity_shared + semantic edges)
  ├── Ego linking (if modifier ∈ {URGENT, DEADLINE, ESCALATION, HIGH_VALUE}
  │                 or relation ∈ {PROPOSAL_SENT, SENT, SCHEDULED, APPROVED, FEEDBACK})
  └── END BATCH (single disk write)
```

**API calls:** 1 (LLM extraction). Up to 3 on retry.

### 3.1 Supersede Protocol

When a new strand has the same primary entity + same relation as an existing
active strand, AND different sentiment, the old strand is marked superseded:
- `old.superseded_by = new.strand_id`
- All outgoing edges from old strand have weights halved

**Risk addressed:** Over-aggressive superseding — see R5 below.

---

## 4. Retrieval Pipeline

```
query(query_text)
  ├── ENCODE: query → CodebookStrand (1 LLM call)
  ├── SEED: find top-3 strands by codebook_similarity (local)
  │     similarity = 0.40 × entity_jaccard
  │                + 0.25 × domain_match
  │                + 0.20 × relation_cluster_sim
  │                + 0.15 × modifier_sim
  ├── ACTIVATE: spreading activation through graph (local)
  │     ├── Adaptive threshold (0.075 - 0.225 based on seed quality + complexity)
  │     ├── Edge type weighting (attention modulation per query type)
  │     ├── Confidence-weighted spreading (0.6 + confidence/5 × 0.4)
  │     ├── Max depth = 4 hops (prevents runaway traversal)
  │     ├── Visited set (prevents re-enqueuing)
  │     └── Recency bonus (warm paths from recent queries)
  ├── BUDGET: select strands within TOKEN_BUDGET=500 tokens
  ├── ASSEMBLE: group by primary entity, sort by timestamp within group
  ├── REASON: feed rendered DNA+traces to LLM (1 LLM call, max_tokens=400)
  └── POST-QUERY LEARNING:
        ├── Increment activation_count for all co-activated strands
        ├── Hebbian update (strengthen edges between co-activated pairs)
        ├── Prime recency buffer
        └── Apply global edge decay (with floor)
```

**API calls:** 2 (encode + reason). Fixed context tokens.

### 4.1 Spreading Activation Detail

```python
queue = deque(seeds)       # BFS with deque (O(1) dequeue)
visited = set(seed_ids)    # prevent re-processing

while queue:
    current_id, current_activation = queue.popleft()
    if depth[current_id] >= MAX_DEPTH:   # cap at 4 hops
        continue

    for neighbor_id, edge_weight, edge_type in graph.neighbors(current_id):
        if neighbor_id in visited:       # skip already-processed
            continue
        spread = current × edge_weight × type_multiplier × confidence × SPREAD_DECAY
        if spread < threshold:
            continue
        activation[neighbor_id] = max(activation.get(neighbor_id, 0), spread)
        visited.add(neighbor_id)
        queue.append((neighbor_id, spread))
```

### 4.2 Attention Modulation

| Query Signal | Edge Boost | Rationale |
|---|---|---|
| temporal=PAST or DEADLINE | temporal × 1.5 | Time-focused recall |
| ≥2 entity slots | entity_shared × 1.3 | Multi-entity queries need entity links |
| domain=GENERAL | causal × 1.4, entity × 1.3 | Cross-domain needs learned + entity paths |
| modifier=URGENT or ESCALATION | ego × 1.5 | Personally significant queries |

---

## 5. Maintenance Operations

### 5.1 Consolidation (Sleep)

Groups active strands by (primary_entity, relation_cluster). Groups with ≥3
strands are merged: most recent kept as "keeper", top 3 traces concatenated,
confidence boosted by 1, older strands superseded.

### 5.2 Intelligent Forgetting

Prunes strands that meet ALL of:
- activation_count ≤ min_activations (default 0)
- age > min_age_seconds (default 30 days)
- NOT ego-linked
- NOT the only strand for their primary entity

### 5.3 Edge Decay

Applied once per query. DECAY_FACTOR = 0.99 with floor = 0.01.
After 100 queries: weight × 0.99^100 ≈ 0.366 of original.
After 500 queries: weight × 0.99^500 ≈ 0.0066 → clamped to 0.01.

---

## 6. Risk Register & Mitigations

### R1: Spreading Activation Unbounded Traversal
- **Risk:** No depth limit or visited set. A well-connected graph causes the
  queue to grow exponentially, re-enqueuing nodes and traversing the entire
  graph. At 10K+ strands this dominates query latency.
- **Impact:** Query time degrades from O(seeds × avg_degree) to O(V + E).
  Activation scores become diluted and meaningless.
- **Mitigation:** Add MAX_DEPTH=4 hop limit. Add visited set to prevent
  re-enqueuing nodes already processed. Nodes can still have their activation
  score upgraded (max semantics) but won't spawn new spreading from the same
  node twice.
- **File:** `expression.py:_spread_activation`

### R2: Entity Alias Match Ignores Entity Type
- **Risk:** Exact alias lookup (`_alias_index` dict) is type-agnostic. If
  "Delta" is first registered as ORG, then a PRODUCT named "Delta" resolves to
  the same instance. This corrupts entity_shared edges — an airline and a
  software product become the same node.
- **Impact:** False entity_shared edges (weight 0.8) create spurious activation
  paths. Retrieval returns unrelated memories.
- **Mitigation:** On exact alias match, verify that the stored entity_type
  matches the requested type. On mismatch, fall through to fuzzy match or
  create a new instance.
- **File:** `entities.py:resolve`

### R3: Cumulative Decay Zeros Out Edges
- **Risk:** `apply_decay()` runs on every query with factor 0.99. After 500
  queries, a weight of 0.8 → 0.005. Edges effectively vanish. Important
  structural connections (entity_shared, ego) silently die.
- **Impact:** Long-running agents lose graph connectivity. Spreading activation
  finds nothing. The system degrades to seed-only retrieval.
- **Mitigation:** Add a weight floor of 0.01. Below this, edges are clamped,
  not decayed further. Ego edges are exempt from decay entirely.
- **File:** `graph.py:apply_decay`

### R4: Seed Finding is O(N) Full-Genome Scan
- **Risk:** `_find_seeds()` iterates every active strand and computes
  codebook_similarity for each. At 100K strands this is 100K comparisons per
  query, each involving set operations and dict lookups.
- **Impact:** Seed phase dominates query latency. At 100K strands, estimated
  50-200ms just for seeding.
- **Mitigation:** Build inverted indexes: entity_id → strand_ids and
  (domain, relation) → strand_ids. Seed candidates = union of strands sharing
  any query entity or domain+relation. Only compute similarity for candidates,
  not entire genome. Falls back to full scan if candidates are empty.
- **File:** `expression.py:_find_seeds`

### R5: Supersede is Over-Aggressive
- **Risk:** Any strand matching same primary_entity + same relation + different
  sentiment triggers supersede. But sentiment legitimately changes over time
  without invalidating the older fact. Example: "Sarah was hesitant about
  pricing" (sentiment=-1) followed by "Sarah re-engaged on pricing"
  (sentiment=+1). The old strand IS still true — it's context, not a
  contradiction.
- **Mitigation:** Only supersede when the relation is in a "belief-state"
  category (e.g., HESITANT, WENT_QUIET, RE_ENGAGED, TRIAL_POSITIVE,
  TRIAL_NEGATIVE) where the newer state genuinely replaces the older one.
  Factual relations (SENT, SCHEDULED, MENTIONED) are never superseded — they
  are historical events.
- **File:** `memory.py:_check_supersede`

### R6: JSON File Corruption on Crash
- **Risk:** `json.dump()` to the same file path is not atomic. If the process
  is killed mid-write, the file is truncated and all data is lost. This applies
  to genome.json, graph.json, and entities.json.
- **Impact:** Total data loss of all memories.
- **Mitigation:** Write to a temporary file first, then atomically rename
  (os.replace). This is atomic on all major filesystems.
- **File:** `genome.py:save`, `graph.py:save`, `entities.py:save`

### R7: Hash-Based Dedup Only Catches Exact Duplicates
- **Risk:** SHA-256 hash of raw_text means even a single character difference
  (trailing space, capitalization) creates a duplicate strand.
- **Impact:** Genome bloat with near-duplicate strands.
- **Severity:** Low. The LLM extraction normalizes semantics into codes, so
  near-duplicates produce similar strands. Consolidation merges them eventually.
- **Mitigation:** Accepted risk. Document as known limitation.

### R8: Codebook is Fixed — No Extension Mechanism
- **Risk:** The 96 codes are hardcoded IntEnums. New domains (e.g., HEALTHCARE)
  or relations (e.g., DIAGNOSED) require code changes and break backwards
  compatibility with existing genome.json files.
- **Impact:** Semantic information loss when interactions don't fit existing
  codes. LLM maps to OTHER/UNKNOWN/GENERAL, degrading retrieval quality.
- **Severity:** Medium. The trace field preserves specifics the codes miss.
- **Mitigation:** Accepted for now. Future: add a dynamic code registration
  layer that maps new codes to the nearest cluster for similarity purposes.

---

## 7. Complexity Analysis

| Operation | Time | Space | Bottleneck |
|---|---|---|---|
| store() | O(E_entity) + 1 LLM call | O(1) strand | Entity edge creation |
| query() — seed | O(candidates) ≤ O(N) | O(N) strand refs | Similarity computation |
| query() — activate | O(V_reachable × avg_degree) | O(V_reachable) | Graph traversal |
| query() — reason | O(1) — 1 LLM call | O(TOKEN_BUDGET) | LLM latency |
| consolidate() | O(N) | O(N) grouping | Full genome scan |
| forget() | O(N) | O(N) entity counts | Full genome scan |
| Graph.add_strand | O(entity_edges + semantic_index) | O(1) per edge | Entity fan-out |

Where N = genome size, E_entity = edges for shared entities.

---

## 8. Persistence Format

### genome.json
```json
[
  {
    "strand_id": "uuid",
    "entity_slots": [[entity_type_int, "instance_id"], ...],
    "relation": 26,
    "modifier": 2,
    "temporal": 2,
    "domain": 0,
    "sentiment": -1,
    "confidence": 4,
    "timestamp": 1711234567,
    "raw_hash": "sha256hex",
    "trace": "Acme Corp $15k budget concerns Q3 deadline",
    "activation_count": 3,
    "superseded_by": null
  }
]
```

### graph.json
```json
{
  "nodes": [{"id": "strand_id", "node_type": "strand"}, {"id": "ego:agent", "node_type": "ego"}],
  "edges": [{"source": "a", "target": "b", "weight": 0.8, "edge_type": "entity_shared", "created": 1711234567}],
  "recency_buffer": [{"id": "strand_id", "warmth": 0.15}]
}
```

### entities.json
```json
[
  {
    "instance_id": "sarah_chen",
    "entity_type": 0,
    "canonical_name": "Sarah Chen",
    "aliases": ["sarah", "sarah chen", "acme's cto"],
    "strand_ids": ["uuid1", "uuid2"],
    "first_seen": 1711234567,
    "last_seen": 1711298765
  }
]
```

---

## 9. Configuration Constants

| Constant | Value | Location | Tuning Notes |
|---|---|---|---|
| SEED_COUNT | 3 | expression.py | More seeds = wider recall, more tokens |
| TOKEN_BUDGET | 500 | expression.py | Must fit in LLM context with prompt |
| BASE_THRESHOLD | 0.15 | expression.py | Lower = more recall, less precision |
| SPREAD_DECAY | 0.7 | expression.py | Per-hop activation decay |
| MAX_DEPTH | 4 | expression.py | Hop limit for activation |
| MAX_EDGES_PER_NODE | 50 | graph.py | Higher = better connectivity, more memory |
| DECAY_FACTOR | 0.99 | graph.py | Per-query edge decay rate |
| DECAY_FLOOR | 0.01 | graph.py | Minimum edge weight after decay |
| HEBBIAN_INCREMENT | 0.15 | graph.py | Co-activation learning rate |
| RECENCY_BONUS | 0.3 | graph.py | Warmth added to recently activated |
| RECENCY_DECAY_RATE | 0.85 | graph.py | Per-cycle recency cooldown |
| Extraction max_tokens | 256 | genome.py | LLM output budget for encoding |
| Reasoning max_tokens | 400 | expression.py | LLM output budget for answers |

---

## 10. Test Coverage

71 unit tests across 5 modules. All tests run without API calls.

| Module | Tests | Coverage |
|---|---|---|
| test_codebook.py | 22 | Encode/decode roundtrips, similarity matrices, clamping |
| test_entities.py | 12 | Alias matching, fuzzy guards, persistence, batch mode |
| test_genome.py | 13 | CRUD, supersede, batch mode, JSON parsing |
| test_graph.py | 13 | Edge types, Hebbian, decay, ego nodes, edge cap |
| test_expression.py | 8 | Similarity function, token estimation |
| test_memory.py | 4 | Consolidation, forgetting, supersede logic |
