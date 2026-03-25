"""
Expression Engine — Brain-like Retrieval Layer (v4 — Brain-Perfect)

The NEUROSCIENCE half of the architecture. Every improvement here mimics
a real brain mechanism:

  1. SEED — find similar strands using code similarity matrix (not just Jaccard)
  2. ACTIVATE — spreading activation with adaptive threshold + confidence weighting
  3. ASSEMBLE — group by entity for coherent narrative (how the brain organizes recall)
  4. REASON — feed DNA codes + neocortical traces directly to LLM

Brain mechanisms implemented:
  - Code similarity matrix (semantic matching, not exact match)
  - Adaptive activation threshold (arousal-dependent gating)
  - Confidence-weighted spreading (certain memories spread stronger)
  - Edge type weighting per query (attention modulation)
  - Query-aware context assembly (narrative coherence)
  - Recency priming (warm paths from recent queries)
  - Activation counting (track which memories are useful)
  - Superseded strand filtering (outdated memories deprioritized)

Total: 2 API calls per query. Fixed context cost. Unlimited genome.
"""

from __future__ import annotations

from collections import defaultdict, deque

import anthropic

from .codebook import (
    Codebook,
    CodebookStrand,
    EntityType,
    Modifier,
    Domain,
    TemporalMarker,
    RelationType,
)
from .config import Config
from .genome import Genome
from .graph import AssociationGraph
from .entities import EntityRegistry


# ─── Brain reasoning prompt ──────────────────────────────────────────────────

def build_reasoning_prompt(codebook: Codebook) -> str:
    """
    System prompt that teaches the LLM to read DNA codes + traces natively.
    Like the brain's innate ability to interpret its own neural patterns.
    """
    entity_legend = ", ".join(
        f"{e.name}={e.value}" for e in sorted(EntityType, key=lambda x: x.value)
        if e.name != "UNKNOWN"
    )
    relation_legend = ", ".join(
        f"{r.name}={r.value}" for r in sorted(RelationType, key=lambda x: x.value)
        if r.name != "OTHER"
    )
    modifier_legend = ", ".join(
        f"{m.name}={m.value}" for m in sorted(Modifier, key=lambda x: x.value)
        if m.name != "OTHER"
    )

    return f"""You are an AI agent with a codebook-encoded memory system. You read compressed memory codes natively.

CODEBOOK:
Entity types: {entity_legend}
Relations: {relation_legend}
Modifiers: {modifier_legend}
Temporal: PAST=0, PRESENT=1, FUTURE=2, RECURRING=3, DEADLINE=4
Domains: SALES=0, TECHNICAL=1, OPS=2, HR=3, FINANCE=4, LEGAL=5, MARKETING=6, SUPPORT=7, GENERAL=8, PRODUCT=9
Sentiment: -2 (very negative) to +2 (very positive)

FORMAT: Each memory has two parts:
  DNA: ENTITY_TYPE:instance_id | REL:relation | MOD:modifier | ...
  Trace: a micro-summary preserving specific facts (numbers, dates, durations)

Memories are grouped by entity and sorted by activation score (higher = more relevant).
Use BOTH the DNA codes (structure/relationships) AND the traces (specific facts) to reason.
Be specific — reference entities, numbers, and facts from the traces. Respond in natural language."""


# ─── Enhanced similarity ─────────────────────────────────────────────────────

def codebook_similarity(
    query: CodebookStrand,
    target: CodebookStrand,
    codebook: Codebook,
) -> float:
    """
    Brain-like similarity using semantic code matching.

    40% — entity instance overlap (Jaccard)
    25% — domain match
    20% — relation similarity (cluster-based, not exact match)
    15% — modifier similarity (sentiment-aligned)
    """
    # Entity overlap via instance IDs
    q_entities = set(query.get_entity_instance_ids())
    t_entities = set(target.get_entity_instance_ids())
    if q_entities or t_entities:
        entity_score = len(q_entities & t_entities) / len(q_entities | t_entities)
    else:
        entity_score = 0.0

    # Domain match
    domain_score = 1.0 if query.domain == target.domain else 0.0

    # Relation similarity (cluster-based — "budget" matches "pricing")
    relation_score = codebook.relation_similarity(query.relation, target.relation)

    # Modifier similarity (sentiment-aligned)
    modifier_score = codebook.modifier_similarity(query.modifier, target.modifier)

    return (
        0.40 * entity_score
        + 0.25 * domain_score
        + 0.20 * relation_score
        + 0.15 * modifier_score
    )


# ─── Token estimation ────────────────────────────────────────────────────────

def estimate_strand_tokens(strand: CodebookStrand) -> int:
    """Estimate tokens for DNA code + trace in the prompt."""
    base = strand.sequence_length() + 10  # DNA code line
    trace_tokens = len(strand.trace.split()) + 2 if strand.trace else 0
    return base + trace_tokens


# ─── Expression Engine ───────────────────────────────────────────────────────

class ExpressionEngine:
    """Brain-perfect retrieval: every mechanism mirrors real neuroscience."""

    def __init__(
        self,
        genome: Genome,
        graph: AssociationGraph,
        codebook: Codebook,
        entity_registry: EntityRegistry,
        model: str = "claude-sonnet-4-20250514",
        config: Config | None = None,
    ):
        self.genome = genome
        self.graph = graph
        self.codebook = codebook
        self.entity_registry = entity_registry
        self.client = anthropic.Anthropic()
        self.model = model
        self._reasoning_prompt = build_reasoning_prompt(codebook)

        cfg = config or Config()
        self.BASE_THRESHOLD = cfg.base_threshold
        self.SPREAD_DECAY = cfg.spread_decay
        self.TOKEN_BUDGET = cfg.token_budget
        self.SEED_COUNT = cfg.seed_count
        self.MAX_DEPTH = cfg.max_spread_depth

    # ── Render DNA + trace ───────────────────────────────────────────────

    def render_strand(self, strand: CodebookStrand) -> str:
        """Render a strand as DNA codes + neocortical trace."""
        parts = []
        for etype, inst_id in strand.entity_slots:
            type_name = self.codebook.decode_entity_type(etype)
            parts.append(f"{type_name}:{inst_id}")

        parts.append(f"REL:{self.codebook.decode_relation(strand.relation)}")
        parts.append(f"MOD:{self.codebook.decode_modifier(strand.modifier)}")
        parts.append(f"TMP:{self.codebook.decode_temporal(strand.temporal)}")
        parts.append(f"DOM:{self.codebook.decode_domain(strand.domain)}")
        parts.append(f"SNT:{strand.sentiment:+d}")

        dna = " | ".join(parts)
        if strand.trace:
            return f"DNA: {dna}\n       Trace: {strand.trace}"
        return f"DNA: {dna}"

    # ── Step 1: SEED (with code similarity matrix) ───────────────────────

    def _find_seeds(self, query_strand: CodebookStrand) -> list[tuple[str, float]]:
        """
        Find seeds using semantic code similarity.

        Uses inverted indexes (entity → strands, domain+relation → strands) to
        narrow candidates before computing similarity. Falls back to full scan
        only if indexes yield no candidates.
        """
        # Build candidate set from indexes
        candidates: set[str] = set()

        # Candidates sharing any entity with the query
        for inst_id in query_strand.get_entity_instance_ids():
            for sid in self.entity_registry.get_strands_for_entity(inst_id):
                candidates.add(sid)

        # Candidates sharing domain+relation (via graph index)
        dr_key = (query_strand.domain, query_strand.relation)
        for sid in self.graph._domain_relation_index.get(dr_key, set()):
            candidates.add(sid)

        # Also check relation cluster neighbors for broader semantic match
        for cluster_id, codes in self.codebook._RELATION_CLUSTERS.items():
            if query_strand.relation in codes:
                for code in codes:
                    cluster_key = (query_strand.domain, code)
                    for sid in self.graph._domain_relation_index.get(cluster_key, set()):
                        candidates.add(sid)
                break

        # Fallback: if indexes yield nothing, scan all active strands
        if not candidates:
            candidates = set(self.genome.active_ids())

        # Extract query keywords for trace-based boosting
        query_text = (query_strand.trace or "").lower()
        query_keywords = set(query_text.split()) - {
            "the", "a", "an", "is", "was", "are", "were", "be", "been",
            "has", "had", "have", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "of", "in",
            "to", "for", "on", "at", "by", "with", "from", "about", "as",
            "and", "or", "but", "not", "no", "if", "it", "its", "this",
            "that", "what", "when", "where", "who", "how", "which", "their",
        }

        scores = []
        for sid in candidates:
            target = self.genome.get(sid)
            if target is None or target.superseded_by is not None:
                continue
            sim = codebook_similarity(query_strand, target, self.codebook)
            if sim > 0:
                # Boost score based on keyword overlap with trace
                if query_keywords and target.trace:
                    trace_lower = target.trace.lower()
                    hits = sum(1 for kw in query_keywords if kw in trace_lower)
                    if hits > 0:
                        # Keyword boost: up to 0.3 extra score
                        boost = min(0.3, hits * 0.1)
                        sim += boost
                scores.append((sid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[: self.SEED_COUNT]

    # ── Step 2: ACTIVATE (adaptive threshold + confidence weighting) ─────

    def _adaptive_threshold(
        self, seeds: list[tuple[str, float]], query_complexity: float
    ) -> float:
        """
        Brain-like arousal gating. The brain lowers its threshold when
        uncertain (need more context) and raises it when confident.
        """
        if not seeds:
            return self.BASE_THRESHOLD * 0.5  # desperate — lower threshold

        avg_seed_score = sum(s for _, s in seeds) / len(seeds)

        if len(seeds) <= 1 and avg_seed_score < 0.4:
            return self.BASE_THRESHOLD * 0.5   # weak seeds — cast wider net
        if len(seeds) >= 3 and avg_seed_score > 0.7:
            return self.BASE_THRESHOLD * 1.5   # strong seeds — be selective
        if query_complexity > 0.7:
            return self.BASE_THRESHOLD * 0.7   # complex query — need more context
        return self.BASE_THRESHOLD

    def _query_edge_weights(self, query_strand: CodebookStrand) -> dict[str, float]:
        """
        Attention modulation — different queries weight edge types differently.
        Like how the brain focuses on different association pathways depending
        on what you're trying to recall.
        """
        weights = {
            "temporal": 1.0, "entity_shared": 1.0, "semantic": 1.0,
            "causal": 1.0, "ego": 1.0,
        }
        # Temporal queries boost temporal edges
        if query_strand.temporal in (TemporalMarker.PAST.value, TemporalMarker.DEADLINE.value):
            weights["temporal"] = 1.5
        # Entity-heavy queries boost entity edges
        if len(query_strand.entity_slots) >= 2:
            weights["entity_shared"] = 1.3
        # Cross-domain queries boost causal edges (learned associations)
        if query_strand.domain == Domain.GENERAL.value:
            weights["causal"] = 1.4
            weights["entity_shared"] = 1.3
        # Ego-significant queries boost ego edges
        if query_strand.modifier in (Modifier.URGENT.value, Modifier.ESCALATION.value):
            weights["ego"] = 1.5
        return weights

    def _spread_activation(
        self,
        seeds: list[tuple[str, float]],
        threshold: float,
        edge_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Spreading activation with:
        - Adaptive threshold (arousal gating)
        - Edge type weighting (attention modulation)
        - Confidence-weighted spreading (certain memories spread stronger)
        - Recency bonus (warm paths)
        - Superseded strand filtering
        """
        activation: dict[str, float] = {}
        depth: dict[str, int] = {}
        visited: set[str] = set()
        queue: deque[tuple[str, float, int]] = deque()  # (id, activation, depth)

        for sid, score in seeds:
            activation[sid] = score
            depth[sid] = 0
            visited.add(sid)
            queue.append((sid, score, 0))

        while queue:
            current_id, current_activation, current_depth = queue.popleft()

            if current_depth >= self.MAX_DEPTH:
                continue

            for neighbor_id, raw_edge_weight, edge_type in self.graph.neighbors(current_id):
                # Skip ego nodes in results
                if neighbor_id.startswith(self.graph.EGO_NODE_PREFIX):
                    continue

                # Skip superseded strands
                neighbor_strand = self.genome.get(neighbor_id)
                if neighbor_strand and neighbor_strand.superseded_by is not None:
                    continue

                # Edge type weighting (attention modulation)
                type_multiplier = edge_weights.get(edge_type, 1.0)

                # Confidence-weighted spreading
                confidence_factor = 1.0
                if neighbor_strand:
                    confidence_factor = 0.6 + (neighbor_strand.confidence / 5.0) * 0.4

                spread = (
                    current_activation
                    * raw_edge_weight
                    * type_multiplier
                    * confidence_factor
                    * self.SPREAD_DECAY
                )

                if spread < threshold:
                    continue

                # Update activation score (always take max)
                if neighbor_id not in activation or spread > activation[neighbor_id]:
                    activation[neighbor_id] = spread

                # Only enqueue if not already visited (prevents re-traversal)
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, spread, current_depth + 1))

        # Add recency bonuses
        for sid in list(activation.keys()):
            bonus = self.graph.get_recency_bonus(sid)
            if bonus > 0:
                activation[sid] += bonus

        return activation

    # ── Step 3: BUDGET + ASSEMBLE (query-aware context) ──────────────────

    def _assemble_context(
        self, activation: dict[str, float]
    ) -> tuple[list[tuple[str, float]], int]:
        """
        Brain-like context assembly:
        1. Budget select within token limit
        2. Group by primary entity for narrative coherence
        3. Sort by temporal order within each group

        Returns (selected strands, total tokens).
        """
        ranked = sorted(activation.items(), key=lambda x: x[1], reverse=True)

        # Budget selection
        selected = []
        total_tokens = 0
        for sid, score in ranked:
            strand = self.genome.get(sid)
            if strand is None:
                continue
            est = estimate_strand_tokens(strand)
            if total_tokens + est > self.TOKEN_BUDGET:
                break
            selected.append((sid, score))
            total_tokens += est

        return selected, total_tokens

    def _group_by_entity(
        self, selected: list[tuple[str, float]]
    ) -> list[tuple[str, list[tuple[str, float, CodebookStrand]]]]:
        """Group activated strands by primary entity for coherent presentation."""
        groups: dict[str, list[tuple[str, float, CodebookStrand]]] = defaultdict(list)

        for sid, score in selected:
            strand = self.genome.get(sid)
            if strand is None:
                continue
            primary = strand.entity_slots[0][1] if strand.entity_slots else "unknown"
            groups[primary].append((sid, score, strand))

        # Sort groups by max activation (most relevant entity first)
        sorted_groups = sorted(
            groups.items(),
            key=lambda g: max(s[1] for s in g[1]),
            reverse=True,
        )

        # Sort strands within each group by timestamp (narrative order)
        for entity, strands in sorted_groups:
            strands.sort(key=lambda s: s[2].timestamp)

        return sorted_groups

    # ── Step 4: REASON (brain reads DNA + traces directly) ───────────────

    def _reason(
        self,
        query_text: str,
        entity_groups: list[tuple[str, list[tuple[str, float, CodebookStrand]]]],
    ) -> str:
        """
        Feed DNA codes + traces to LLM, grouped by entity.
        The brain reads its own patterns natively. 1 API call.
        """
        memory_lines = []
        for entity, strands in entity_groups:
            memory_lines.append(f"\n  --- {entity} ---")
            for sid, score, strand in strands:
                rendered = self.render_strand(strand)
                memory_lines.append(f"  [{score:.2f}] {rendered}")

        memory_block = "\n".join(memory_lines)

        user_message = f"""Active memories (grouped by entity, sorted by relevance):
{memory_block}

Question: {query_text}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            system=self._reasoning_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()

    # ── Query complexity estimation ──────────────────────────────────────

    def _estimate_query_complexity(self, query_strand: CodebookStrand) -> float:
        """Heuristic complexity from query structure."""
        score = 0.3
        score += len(query_strand.entity_slots) * 0.1
        if query_strand.modifier == Modifier.UNCERTAIN.value:
            score += 0.2
        if query_strand.domain == Domain.GENERAL.value:
            score += 0.15
        # Comparative queries (multiple entities) are complex
        if len(query_strand.entity_slots) >= 2:
            score += 0.15
        return min(1.0, score)

    # ── Full expression pipeline ─────────────────────────────────────────

    def express(self, query_text: str, query_strand: CodebookStrand) -> dict:
        """
        Full brain-perfect expression pipeline:
          1. SEED     — semantic code similarity (not just Jaccard)
          2. ACTIVATE — adaptive threshold + confidence weighting + edge modulation
          3. ASSEMBLE — budget + group by entity for narrative coherence
          4. REASON   — LLM reads DNA + traces directly (1 API call)

        Total: 1 API call for reasoning. Fixed context cost.
        """
        # Query analysis
        query_complexity = self._estimate_query_complexity(query_strand)

        # Step 1: Seeds (semantic matching)
        seeds = self._find_seeds(query_strand)

        # Step 2: Spreading activation (adaptive + confidence + edge-weighted)
        threshold = self._adaptive_threshold(seeds, query_complexity)
        edge_weights = self._query_edge_weights(query_strand)
        activation = self._spread_activation(seeds, threshold, edge_weights)

        # Step 3: Budget + assemble context (entity-grouped narrative)
        selected, tokens_used = self._assemble_context(activation)
        entity_groups = self._group_by_entity(selected)

        # Step 4: Brain-like reasoning (1 API call)
        answer = self._reason(query_text, entity_groups)

        # Post-query learning
        co_activated_ids = [sid for sid, _ in selected]

        # Track activation counts
        for sid in co_activated_ids:
            self.genome.increment_activation(sid)

        # Hebbian learning
        if len(co_activated_ids) > 1:
            self.graph.hebbian_update(co_activated_ids)

        # Recency priming
        self.graph.prime_recency(co_activated_ids)
        self.graph.decay_recency()

        # Decay
        self.graph.apply_decay()

        # Build activated details for reporting
        activated_details = []
        for sid, score in selected:
            strand = self.genome.get(sid)
            if strand is None:
                continue
            dna_code = self.render_strand(strand)
            activated_details.append((sid, round(score, 3), dna_code))

        # Not activated
        all_ids = set(self.genome.active_ids())
        activated_ids = {sid for sid, _, _ in activated_details}
        not_activated = list(all_ids - activated_ids)

        # Naive cost
        tokens_naive = self.genome.count() * 33

        return {
            "answer": answer,
            "activated": activated_details,
            "not_activated": not_activated,
            "tokens_used": tokens_used,
            "tokens_naive": tokens_naive,
            "api_calls": 1,
            "threshold_used": round(threshold, 3),
            "query_complexity": round(query_complexity, 2),
            "entity_groups": len(entity_groups),
        }
