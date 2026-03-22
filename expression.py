"""
Expression Engine — Layer 3 of the Cognitive Memory Architecture

Implements spreading activation retrieval. This layer NEVER loads all strands
into context. It:
  1. SEEDs from user input (temporary strand → find closest existing strands)
  2. TRAVERSEs the association graph via spreading activation
  3. BUDGETs activated nodes to a fixed token limit
  4. DECODEs selected strands back into natural language

Key property: context window token cost stays FIXED regardless of genome size.
"""

import anthropic

from genome import Genome
from graph import AssociationGraph


# ─── Similarity scoring ─────────────────────────────────────────────────────

def entity_overlap_score(a_entities: list[str], b_entities: list[str]) -> float:
    """Jaccard-like overlap on entity sets."""
    a_set = {e.lower() for e in a_entities}
    b_set = {e.lower() for e in b_entities}
    if not a_set and not b_set:
        return 0.0
    intersection = a_set & b_set
    union = a_set | b_set
    return len(intersection) / len(union)


def strand_similarity(query_strand: dict, target_strand: dict) -> float:
    """
    Compute similarity between a query strand and a stored strand.
    Uses entity overlap (70%) + domain match (30%).
    """
    entity_score = entity_overlap_score(
        query_strand["encoded"]["entities"],
        target_strand["encoded"]["entities"],
    )
    domain_score = 1.0 if query_strand["encoded"]["domain"] == target_strand["encoded"]["domain"] else 0.0
    return 0.7 * entity_score + 0.3 * domain_score


# ─── Token estimation ────────────────────────────────────────────────────────

def estimate_strand_tokens(strand: dict) -> int:
    """Estimate how many tokens a decoded strand will cost in context."""
    encoded = strand["encoded"]
    # Rough estimate: entities + relation + value + metadata ≈ 20-30 tokens per strand
    text_len = (
        sum(len(e) for e in encoded["entities"])
        + len(encoded["relation"])
        + len(encoded["value"])
        + len(encoded["domain"])
    )
    return max(15, text_len // 3)


def estimate_raw_tokens(raw_text: str) -> int:
    """Estimate tokens for raw text (naive approach without encoding)."""
    return max(1, len(raw_text.split()) * 1.3).__ceil__()


# ─── Decoder ─────────────────────────────────────────────────────────────────

DECODE_SYSTEM = """You are a memory decoder. Given a compressed memory strand (JSON),
produce a natural language summary in 1-2 sentences. Be concise and factual.
Return ONLY the summary text, no JSON, no markdown."""


class ExpressionEngine:
    """Spreading activation retrieval over the memory genome."""

    ACTIVATION_THRESHOLD = 0.15
    SPREAD_DECAY = 0.7           # activation *= edge_weight * SPREAD_DECAY
    TOKEN_BUDGET = 400           # max tokens to use in context
    SEED_COUNT = 3               # number of seed nodes

    def __init__(
        self,
        genome: Genome,
        graph: AssociationGraph,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.genome = genome
        self.graph = graph
        self.client = anthropic.Anthropic()
        self.model = model

    # ── Step 1: SEED ─────────────────────────────────────────────────────

    def _find_seeds(self, query_strand: dict) -> list[tuple[str, float]]:
        """Find the top-N most similar strands to the query."""
        scores = []
        for sid in self.genome.all_ids():
            target = self.genome.get(sid)
            if target is None:
                continue
            sim = strand_similarity(query_strand, target)
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

        # Initialize seeds
        for sid, score in seeds:
            activation[sid] = score
            queue.append((sid, score))

        # BFS-like spreading
        while queue:
            current_id, current_activation = queue.pop(0)

            for neighbor_id, edge_weight, _ in self.graph.neighbors(current_id):
                spread = current_activation * edge_weight * self.SPREAD_DECAY

                if spread < self.ACTIVATION_THRESHOLD:
                    continue

                # Only spread if this gives a higher activation than existing
                if neighbor_id not in activation or spread > activation[neighbor_id]:
                    activation[neighbor_id] = spread
                    queue.append((neighbor_id, spread))

        return activation

    # ── Step 3: BUDGET ───────────────────────────────────────────────────

    def _budget_select(self, activation: dict[str, float]) -> list[tuple[str, float]]:
        """Select top activated strands within token budget."""
        ranked = sorted(activation.items(), key=lambda x: x[1], reverse=True)

        selected = []
        total_tokens = 0

        for sid, score in ranked:
            strand = self.genome.get(sid)
            if strand is None:
                continue
            est_tokens = estimate_strand_tokens(strand)
            if total_tokens + est_tokens > self.TOKEN_BUDGET:
                break
            selected.append((sid, score))
            total_tokens += est_tokens

        return selected

    # ── Step 4: DECODE ───────────────────────────────────────────────────

    def _decode_strand(self, strand: dict) -> str:
        """Decode a compressed strand back into natural language."""
        import json as _json

        encoded_str = _json.dumps(strand["encoded"], indent=2)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            system=DECODE_SYSTEM,
            messages=[{"role": "user", "content": encoded_str}],
        )
        return response.content[0].text.strip()

    # ── Full expression pipeline ─────────────────────────────────────────

    def express(self, query_strand: dict) -> dict:
        """
        Full expression pipeline: seed → traverse → budget → decode.

        Returns:
            {
                "activated": [(strand_id, activation_score, decoded_text), ...],
                "not_activated": [strand_id, ...],
                "tokens_used": int,
                "tokens_naive": int,
            }
        """
        # Step 1: Find seeds
        seeds = self._find_seeds(query_strand)

        # Step 2: Spreading activation
        activation = self._spread_activation(seeds)

        # Step 3: Budget selection
        selected = self._budget_select(activation)

        # Step 4: Decode selected strands
        activated_results = []
        tokens_used = 0
        for sid, score in selected:
            strand = self.genome.get(sid)
            if strand is None:
                continue
            decoded = self._decode_strand(strand)
            est_tokens = estimate_strand_tokens(strand)
            tokens_used += est_tokens
            activated_results.append((sid, round(score, 3), decoded))

        # Hebbian update: co-activated strands strengthen edges
        co_activated_ids = [sid for sid, _, _ in activated_results]
        if len(co_activated_ids) > 1:
            self.graph.hebbian_update(co_activated_ids)

        # Apply decay
        self.graph.apply_decay()

        # Compute what was NOT activated
        all_ids = set(self.genome.all_ids())
        activated_ids = {sid for sid, _, _ in activated_results}
        not_activated = list(all_ids - activated_ids)

        # Naive token cost estimate (if we dumped all raw memories)
        # Assume ~25 words per raw interaction ≈ ~33 tokens each
        tokens_naive = self.genome.count() * 33

        return {
            "activated": activated_results,
            "not_activated": not_activated,
            "tokens_used": tokens_used,
            "tokens_naive": tokens_naive,
        }
