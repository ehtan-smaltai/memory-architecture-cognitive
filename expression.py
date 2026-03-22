"""
Expression Engine — Brain-like Retrieval Layer

This is the NEUROSCIENCE half of the architecture. The molecular biology
half (Protein → RNA → DNA) handles storage. This layer handles retrieval
using brain-like mechanisms:

  1. ENCODE  — query → DNA codes (1 API call)
  2. ACTIVATE — spreading activation through graph (LOCAL, 0 API calls)
  3. REASON  — feed activated DNA codes directly to LLM (1 API call)

The LLM reads codebook sequences natively — just like the brain doesn't
"decompress" neural activation patterns before reasoning. The pattern
of activation IS the understanding.

Total retrieval cost: 2 API calls + ~200-400 context tokens, regardless
of genome size. No decode step. No intermediate reconstruction.
"""

from __future__ import annotations

import anthropic

from codebook import Codebook, CodebookStrand, Modifier, Domain
from genome import Genome
from graph import AssociationGraph
from entities import EntityRegistry


# ─── Brain reasoning prompt ──────────────────────────────────────────────────

def build_reasoning_prompt(codebook: Codebook) -> str:
    """
    System prompt that teaches the LLM to read DNA codes directly.
    Loaded once per session — like the brain's innate ability to
    interpret its own neural patterns.
    """
    entity_legend = ", ".join(
        f"{e.name}={e.value}" for e in sorted(
            __import__('codebook').EntityType, key=lambda x: x.value
        ) if e.name != "UNKNOWN"
    )
    relation_legend = ", ".join(
        f"{r.name}={r.value}" for r in sorted(
            __import__('codebook').RelationType, key=lambda x: x.value
        ) if r.name != "OTHER"
    )
    modifier_legend = ", ".join(
        f"{m.name}={m.value}" for m in sorted(
            __import__('codebook').Modifier, key=lambda x: x.value
        ) if m.name != "OTHER"
    )

    return f"""You are an AI agent with a codebook-encoded memory system. You can read compressed memory codes directly — no translation needed.

CODEBOOK LEGEND:
Entity types: {entity_legend}
Relations: {relation_legend}
Modifiers: {modifier_legend}
Temporal: PAST=0, PRESENT=1, FUTURE=2, RECURRING=3, DEADLINE=4
Domains: SALES=0, TECHNICAL=1, OPS=2, HR=3, FINANCE=4, LEGAL=5, MARKETING=6, SUPPORT=7, GENERAL=8, PRODUCT=9
Sentiment: -2 (very negative) to +2 (very positive)
Confidence: 1 (uncertain) to 5 (certain)

FORMAT: Each memory is shown as:
  [activation_score] ENTITY_TYPE:instance_id | ... | REL:relation | MOD:modifier | TMP:temporal | DOM:domain | SNT:sentiment

You understand these codes natively. Reason over them directly to answer the user's question. Be specific, reference the entities and relationships you see in the memories. Respond in natural language."""


# ─── Similarity for codebook strands ─────────────────────────────────────────

def codebook_similarity(query: CodebookStrand, target: CodebookStrand) -> float:
    """
    Compute similarity between two CodebookStrands.
    40% entity overlap + 25% domain + 20% relation + 15% modifier.
    """
    q_entities = set(query.get_entity_instance_ids())
    t_entities = set(target.get_entity_instance_ids())
    if q_entities or t_entities:
        entity_score = len(q_entities & t_entities) / len(q_entities | t_entities)
    else:
        entity_score = 0.0

    domain_score = 1.0 if query.domain == target.domain else 0.0
    relation_score = 1.0 if query.relation == target.relation else 0.0
    modifier_score = 1.0 if query.modifier == target.modifier else 0.0

    return (
        0.40 * entity_score
        + 0.25 * domain_score
        + 0.20 * relation_score
        + 0.15 * modifier_score
    )


# ─── Token estimation ────────────────────────────────────────────────────────

def estimate_dna_tokens(strand: CodebookStrand) -> int:
    """Estimate tokens for a DNA code line in the prompt."""
    # Each strand renders as ~15-25 tokens in code format
    return strand.sequence_length() + 10


# ─── Expression Engine ───────────────────────────────────────────────────────

class ExpressionEngine:
    """Brain-like retrieval: activate → reason directly over DNA codes."""

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
        self._reasoning_prompt = build_reasoning_prompt(codebook)

    # ── Render DNA codes ─────────────────────────────────────────────────

    def render_dna(self, strand: CodebookStrand) -> str:
        """
        Render a strand as a human/LLM-readable DNA code string.
        This is what gets fed directly into the LLM prompt.
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

    # ── Step 1: SEED ─────────────────────────────────────────────────────

    def _find_seeds(self, query_strand: CodebookStrand) -> list[tuple[str, float]]:
        """Find top-N most similar strands to the query."""
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

    # ── Step 2: ACTIVATE (Spreading Activation — LOCAL, no API) ──────────

    def _spread_activation(self, seeds: list[tuple[str, float]]) -> dict[str, float]:
        """
        Run spreading activation from seed nodes through the graph.
        This is LOCAL computation — zero API calls, zero network latency.
        Just like neural activation in the brain.
        """
        activation: dict[str, float] = {}
        queue: list[tuple[str, float]] = []

        for sid, score in seeds:
            activation[sid] = score
            queue.append((sid, score))

        while queue:
            current_id, current_activation = queue.pop(0)

            for neighbor_id, edge_weight, _ in self.graph.neighbors(current_id):
                if neighbor_id.startswith(self.graph.EGO_NODE_PREFIX):
                    continue

                spread = current_activation * edge_weight * self.SPREAD_DECAY

                if spread < self.ACTIVATION_THRESHOLD:
                    continue

                if neighbor_id not in activation or spread > activation[neighbor_id]:
                    activation[neighbor_id] = spread
                    queue.append((neighbor_id, spread))

        # Add recency bonuses (warm paths from recent queries)
        for sid in list(activation.keys()):
            bonus = self.graph.get_recency_bonus(sid)
            if bonus > 0:
                activation[sid] += bonus

        return activation

    # ── Step 3: BUDGET — select within token limit ───────────────────────

    def _budget_select(
        self, activation: dict[str, float]
    ) -> list[tuple[str, float]]:
        """Select top activated strands within token budget."""
        ranked = sorted(activation.items(), key=lambda x: x[1], reverse=True)

        selected = []
        total_tokens = 0

        for sid, score in ranked:
            strand = self.genome.get(sid)
            if strand is None:
                continue

            est_tokens = estimate_dna_tokens(strand)
            if total_tokens + est_tokens > self.TOKEN_BUDGET:
                break

            selected.append((sid, score))
            total_tokens += est_tokens

        return selected

    # ── Step 4: REASON — LLM reads DNA codes directly ───────────────────

    def _reason(
        self,
        query_text: str,
        activated_strands: list[tuple[str, float]],
    ) -> str:
        """
        Feed activated DNA codes directly to the LLM for reasoning.
        The LLM reads codebook sequences natively — no decode step.
        This is the brain-like part: the activation pattern IS the input.

        1 API call. That's it.
        """
        # Build the memory context from DNA codes
        memory_lines = []
        for sid, score in activated_strands:
            strand = self.genome.get(sid)
            if strand is None:
                continue
            dna_code = self.render_dna(strand)
            memory_lines.append(f"  [{score:.2f}] {dna_code}")

        memory_block = "\n".join(memory_lines)

        user_message = f"""Active memories (sorted by relevance):
{memory_block}

Question: {query_text}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=self._reasoning_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()

    # ── Full expression pipeline ─────────────────────────────────────────

    def express(self, query_text: str, query_strand: CodebookStrand) -> dict:
        """
        Full brain-like expression pipeline:
          1. SEED     — find similar strands (local)
          2. ACTIVATE — spreading activation (local, 0 API calls)
          3. BUDGET   — select within token limit (local)
          4. REASON   — LLM reads DNA codes directly (1 API call)

        Total: 1 API call for reasoning + the 1 API call for query encoding
        that already happened in memory.py = 2 API calls total.
        """
        # Step 1: Seeds
        seeds = self._find_seeds(query_strand)

        # Step 2: Spreading activation (LOCAL — no API calls)
        activation = self._spread_activation(seeds)

        # Step 3: Budget selection (LOCAL)
        selected = self._budget_select(activation)

        # Step 4: Brain-like reasoning (1 API call)
        answer = self._reason(query_text, selected)

        # Post-query: Hebbian update + recency priming
        co_activated_ids = [sid for sid, _ in selected]
        if len(co_activated_ids) > 1:
            self.graph.hebbian_update(co_activated_ids)
        self.graph.prime_recency(co_activated_ids)
        self.graph.decay_recency()
        self.graph.apply_decay()

        # Build activated details for reporting
        activated_details = []
        tokens_used = 0
        for sid, score in selected:
            strand = self.genome.get(sid)
            if strand is None:
                continue
            dna_code = self.render_dna(strand)
            est_tokens = estimate_dna_tokens(strand)
            tokens_used += est_tokens
            activated_details.append((sid, round(score, 3), dna_code))

        # Not activated
        all_ids = set(self.genome.all_ids())
        activated_ids = {sid for sid, _, _ in activated_details}
        not_activated = list(all_ids - activated_ids)

        # Naive cost comparison
        tokens_naive = self.genome.count() * 33

        return {
            "answer": answer,
            "activated": activated_details,   # (strand_id, score, dna_code)
            "not_activated": not_activated,
            "tokens_used": tokens_used,
            "tokens_naive": tokens_naive,
            "api_calls": 1,                   # just the reasoning call
        }
