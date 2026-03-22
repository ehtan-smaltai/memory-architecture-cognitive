"""
Expression Engine — Layer 3 of the Cognitive Memory Architecture (v2)

Implements DNA → RNA → Protein multi-resolution decoding:
  DNA level:     Codebook codes rendered as readable string (0 API calls)
  RNA level:     Structured 1-sentence summary (1 cheap API call)
  PROTEIN level: Full natural language reconstruction (1 full API call)

The engine probes at the cheapest level first and only escalates when needed,
inspired by Recursive Language Models (arXiv 2512.24601): treat memory as an
external environment, probe recursively, only decode what's necessary.

Key property: context window token cost stays FIXED regardless of genome size.
"""

from __future__ import annotations

from enum import IntEnum

import anthropic

from codebook import (
    Codebook,
    CodebookStrand,
    Modifier,
    Domain,
)
from genome import Genome
from graph import AssociationGraph
from entities import EntityRegistry


# ─── Decode levels ───────────────────────────────────────────────────────────

class DecodeLevel(IntEnum):
    DNA = 0      # codebook codes — FREE (no API call)
    RNA = 1      # structured summary — cheap API call
    PROTEIN = 2  # full natural language — full API call


# ─── Decode prompts ──────────────────────────────────────────────────────────

RNA_DECODE_SYSTEM = """You are a memory decoder. Given a codebook-encoded memory strand,
produce a BRIEF structured summary in 1 sentence. Include the key entity, action, and context.
Max 30 words. Return ONLY the summary text, no JSON, no markdown."""

PROTEIN_DECODE_SYSTEM = """You are a memory decoder. Given a codebook-encoded memory strand,
reconstruct the full context as natural language. Include all entities, relationships,
sentiment, timing, and implications. 2-3 sentences max. Return ONLY the text, no JSON, no markdown."""


# ─── Similarity for codebook strands ─────────────────────────────────────────

def codebook_similarity(query: CodebookStrand, target: CodebookStrand) -> float:
    """
    Compute similarity between two CodebookStrands using codebook codes.

    Scoring (weighted):
      40% — entity instance overlap (Jaccard on instance_ids)
      25% — domain match
      20% — relation match
      15% — modifier match
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

    # Relation match
    relation_score = 1.0 if query.relation == target.relation else 0.0

    # Modifier match
    modifier_score = 1.0 if query.modifier == target.modifier else 0.0

    return (
        0.40 * entity_score
        + 0.25 * domain_score
        + 0.20 * relation_score
        + 0.15 * modifier_score
    )


# ─── Token estimation ────────────────────────────────────────────────────────

def estimate_strand_tokens(strand: CodebookStrand, level: DecodeLevel) -> int:
    """Estimate token cost based on decode level."""
    if level == DecodeLevel.DNA:
        return strand.sequence_length() + 5  # codes + separators
    elif level == DecodeLevel.RNA:
        return 35  # ~30 words avg
    else:  # PROTEIN
        return 75  # ~60-100 words avg


# ─── Expression Engine ───────────────────────────────────────────────────────

class ExpressionEngine:
    """Spreading activation retrieval with multi-resolution decoding."""

    ACTIVATION_THRESHOLD = 0.15
    SPREAD_DECAY = 0.7
    TOKEN_BUDGET = 400
    SEED_COUNT = 3

    def __init__(
        self,
        genome: Genome,
        graph: AssociationGraph,
        codebook: Codebook,
        entity_registry: EntityRegistry,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.genome = genome
        self.graph = graph
        self.codebook = codebook
        self.entity_registry = entity_registry
        self.client = anthropic.Anthropic()
        self.model = model

    # ── Step 1: SEED ─────────────────────────────────────────────────────

    def _find_seeds(self, query_strand: CodebookStrand) -> list[tuple[str, float]]:
        """Find the top-N most similar strands to the query."""
        scores = []
        for sid in self.genome.all_ids():
            target = self.genome.get(sid)
            if target is None:
                continue
            sim = codebook_similarity(query_strand, target)
            if sim > 0:
                scores.append((sid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[: self.SEED_COUNT]

    # ── Step 2: TRAVERSE (Spreading Activation) ─────────────────────────

    def _spread_activation(self, seeds: list[tuple[str, float]]) -> dict[str, float]:
        """
        Run spreading activation from seed nodes through the graph.
        Returns {strand_id: activation_score} for all activated nodes.
        """
        activation: dict[str, float] = {}
        queue: list[tuple[str, float]] = []

        for sid, score in seeds:
            activation[sid] = score
            queue.append((sid, score))

        while queue:
            current_id, current_activation = queue.pop(0)

            for neighbor_id, edge_weight, _ in self.graph.neighbors(current_id):
                # Skip ego nodes in activation results
                if neighbor_id.startswith(self.graph.EGO_NODE_PREFIX):
                    continue

                spread = current_activation * edge_weight * self.SPREAD_DECAY

                if spread < self.ACTIVATION_THRESHOLD:
                    continue

                if neighbor_id not in activation or spread > activation[neighbor_id]:
                    activation[neighbor_id] = spread
                    queue.append((neighbor_id, spread))

        # Add recency bonuses
        for sid in list(activation.keys()):
            bonus = self.graph.get_recency_bonus(sid)
            if bonus > 0:
                activation[sid] += bonus

        return activation

    # ── Step 3: BUDGET ───────────────────────────────────────────────────

    def _budget_select(
        self, activation: dict[str, float], query_complexity: float
    ) -> list[tuple[str, float, DecodeLevel]]:
        """Select top activated strands within token budget, choosing decode level."""
        ranked = sorted(activation.items(), key=lambda x: x[1], reverse=True)

        selected = []
        total_tokens = 0

        for sid, score in ranked:
            strand = self.genome.get(sid)
            if strand is None:
                continue

            level = self._choose_decode_level(score, query_complexity)
            est_tokens = estimate_strand_tokens(strand, level)

            if total_tokens + est_tokens > self.TOKEN_BUDGET:
                break

            selected.append((sid, score, level))
            total_tokens += est_tokens

        return selected

    # ── Step 4: DECODE (DNA → RNA → Protein) ─────────────────────────────

    def _decode_dna(self, strand: CodebookStrand) -> str:
        """
        Level 0: Render codebook codes as human-readable string.
        ZERO API calls — this is FREE.
        """
        parts = []
        for etype, inst_id in strand.entity_slots:
            type_name = self.codebook.decode_entity_type(etype)
            parts.append(f"{type_name}:{inst_id}")

        parts.append(f"REL:{self.codebook.decode_relation(strand.relation)}")
        parts.append(f"MOD:{self.codebook.decode_modifier(strand.modifier)}")
        parts.append(f"TMP:{self.codebook.decode_temporal(strand.temporal)}")
        parts.append(f"DOM:{self.codebook.decode_domain(strand.domain)}")
        parts.append(f"SNT:{strand.sentiment:+d}")
        return " | ".join(parts)

    def _decode_rna(self, strand: CodebookStrand) -> str:
        """Level 1: Structured 1-sentence summary via cheap API call."""
        dna_text = self._decode_dna(strand)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=60,
            system=RNA_DECODE_SYSTEM,
            messages=[{"role": "user", "content": dna_text}],
        )
        return response.content[0].text.strip()

    def _decode_protein(self, strand: CodebookStrand) -> str:
        """Level 2: Full natural language reconstruction via API call."""
        dna_text = self._decode_dna(strand)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,
            system=PROTEIN_DECODE_SYSTEM,
            messages=[{"role": "user", "content": dna_text}],
        )
        return response.content[0].text.strip()

    def _choose_decode_level(
        self, activation_score: float, query_complexity: float
    ) -> DecodeLevel:
        """
        RLM-inspired: probe at cheapest level first, escalate only when needed.

        High activation + complex query → PROTEIN (full decode)
        Moderate activation or complexity → RNA (structured summary)
        Low activation → DNA (just codes, free)
        """
        if activation_score >= 0.7 and query_complexity >= 0.6:
            return DecodeLevel.PROTEIN
        elif activation_score >= 0.4 or query_complexity >= 0.4:
            return DecodeLevel.RNA
        else:
            return DecodeLevel.DNA

    def _estimate_query_complexity(self, query_strand: CodebookStrand) -> float:
        """
        Heuristic: queries with more entities, uncertain modifiers,
        or cross-domain scope are more complex.
        """
        score = 0.3  # base
        score += len(query_strand.entity_slots) * 0.1
        if query_strand.modifier == Modifier.UNCERTAIN.value:
            score += 0.2
        if query_strand.domain == Domain.GENERAL.value:
            score += 0.15
        return min(1.0, score)

    # ── Full expression pipeline ─────────────────────────────────────────

    def express(self, query_strand: CodebookStrand) -> dict:
        """
        Full expression pipeline: seed → traverse → budget → decode.

        Returns:
            {
                "activated": [(strand_id, score, decoded_text, level_name), ...],
                "not_activated": [strand_id, ...],
                "tokens_used": int,
                "tokens_naive": int,
                "api_calls_made": int,
                "api_calls_saved": int,
                "decode_levels": {"DNA": n, "RNA": n, "PROTEIN": n},
            }
        """
        # Step 1: Seeds
        seeds = self._find_seeds(query_strand)

        # Step 2: Spreading activation
        activation = self._spread_activation(seeds)

        # Step 3: Query complexity + budget selection
        query_complexity = self._estimate_query_complexity(query_strand)
        selected = self._budget_select(activation, query_complexity)

        # Step 4: Multi-resolution decode
        activated_results = []
        tokens_used = 0
        api_calls = 0
        level_counts = {"DNA": 0, "RNA": 0, "PROTEIN": 0}

        for sid, score, level in selected:
            strand = self.genome.get(sid)
            if strand is None:
                continue

            if level == DecodeLevel.DNA:
                decoded = self._decode_dna(strand)
            elif level == DecodeLevel.RNA:
                decoded = self._decode_rna(strand)
                api_calls += 1
            else:
                decoded = self._decode_protein(strand)
                api_calls += 1

            est_tokens = estimate_strand_tokens(strand, level)
            tokens_used += est_tokens
            level_counts[level.name] += 1
            activated_results.append((sid, round(score, 3), decoded, level.name))

        # Hebbian update
        co_activated_ids = [sid for sid, _, _, _ in activated_results]
        if len(co_activated_ids) > 1:
            self.graph.hebbian_update(co_activated_ids)

        # Recency priming
        self.graph.prime_recency(co_activated_ids)
        self.graph.decay_recency()

        # Decay
        self.graph.apply_decay()

        # Not activated
        all_ids = set(self.genome.all_ids())
        activated_ids = {sid for sid, _, _, _ in activated_results}
        not_activated = list(all_ids - activated_ids)

        # Naive cost
        tokens_naive = self.genome.count() * 33

        # API calls saved = total activated decoded at DNA level
        api_calls_saved = level_counts["DNA"]

        return {
            "activated": activated_results,
            "not_activated": not_activated,
            "tokens_used": tokens_used,
            "tokens_naive": tokens_naive,
            "api_calls_made": api_calls,
            "api_calls_saved": api_calls_saved,
            "decode_levels": level_counts,
        }
