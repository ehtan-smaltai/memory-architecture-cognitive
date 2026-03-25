"""
MemorySystem — Brain-Perfect Cognitive Memory (v4)

Two biological systems, fully implemented:

  MOLECULAR BIOLOGY (Storage):
    Protein -> RNA -> DNA compression with neocortical trace

  NEUROSCIENCE (Retrieval):
    Spreading activation + brain reads DNA + traces directly

Brain mechanisms:
  - Codebook encoding (hippocampal indexing)
  - Neocortical trace (fact preservation)
  - Entity normalization (semantic memory)
  - Ego nodes (self-referential memory)
  - Hebbian learning (co-activation strengthening)
  - Recency priming (recent memory boost)
  - Memory consolidation (sleep — merge related memories)
  - Intelligent forgetting (prune unused memories)
  - Strand versioning (supersede outdated info)
  - Confidence-weighted activation (certain memories spread stronger)
  - Adaptive threshold (arousal-dependent gating)
  - Edge type modulation (attention control)
  - Query-aware context assembly (narrative coherence)
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict

from .codebook import Codebook, CodebookStrand, Modifier, RelationType
from .config import Config
from .entities import EntityRegistry
from .genome import DNAEncoder, Genome
from .graph import AssociationGraph
from .expression import ExpressionEngine


class MemorySystem:
    """Brain-perfect cognitive memory architecture."""

    # Relations that indicate the agent took action (bidirectional ego)
    AGENT_ACTION_RELATIONS = {
        RelationType.PROPOSAL_SENT.value,
        RelationType.SENT.value,
        RelationType.SCHEDULED.value,
        RelationType.APPROVED.value,
        RelationType.FEEDBACK.value,
    }

    # Modifiers that indicate personal significance
    EGO_MODIFIERS = {
        Modifier.URGENT.value,
        Modifier.DEADLINE.value,
        Modifier.ESCALATION.value,
        Modifier.HIGH_VALUE.value,
    }

    # Relations representing mutable belief states — newer replaces older.
    # Factual/event relations (SENT, SCHEDULED, MENTIONED, etc.) are NOT
    # supersedable because they represent historical events, not current state.
    SUPERSEDABLE_RELATIONS = {
        RelationType.HESITANT.value,
        RelationType.WENT_QUIET.value,
        RelationType.RE_ENGAGED.value,
        RelationType.TRIAL_POSITIVE.value,
        RelationType.TRIAL_NEGATIVE.value,
        RelationType.TRIAL_STARTED.value,
        RelationType.PRICE_CONCERN.value,
        RelationType.EXPANDING.value,
        RelationType.BREAKING_DOWN.value,
        RelationType.COMPETING.value,
        RelationType.CANCELLED.value,
        RelationType.RENEWED.value,
    }

    def __init__(
        self,
        config: Config | None = None,
        *,
        # Legacy kwargs for backwards compatibility
        genome_path: str | None = None,
        graph_path: str | None = None,
        entities_path: str | None = None,
        model: str | None = None,
    ):
        cfg = config or Config()
        # Allow legacy kwargs to override config
        if genome_path is not None:
            cfg.genome_path = genome_path
        if graph_path is not None:
            cfg.graph_path = graph_path
        if entities_path is not None:
            cfg.entities_path = entities_path
        if model is not None:
            cfg.model = model

        self.config = cfg
        self.codebook = Codebook()
        self.entity_registry = EntityRegistry(path=cfg.entities_path)
        self.encoder = DNAEncoder(
            codebook=self.codebook,
            entity_registry=self.entity_registry,
            model=cfg.model,
            max_retries=cfg.max_extraction_retries,
            max_entities=cfg.max_entities_per_strand,
        )
        self.genome = Genome(path=cfg.genome_path)
        self.graph = AssociationGraph(path=cfg.graph_path, config=cfg)
        self.expression = ExpressionEngine(
            genome=self.genome,
            graph=self.graph,
            codebook=self.codebook,
            entity_registry=self.entity_registry,
            model=cfg.model,
            config=cfg,
        )
        self._recent_ids: list[str] = list(self.genome.all_ids())
        self.graph.ensure_ego_node("agent")
        # Rebuild semantic index from persisted genome
        self.graph.rebuild_domain_relation_index(self.genome.get)

    # ── Storage Pipeline (Molecular Biology) ─────────────────────────────

    def store(self, raw_text: str, timestamp: int | None = None) -> CodebookStrand | None:
        """
        Protein -> RNA -> DNA storage with neocortical trace.

        Now includes:
        - Trace extraction (micro-summary preserving key facts)
        - Bidirectional ego linking (actions + significant events)
        - Strand versioning (supersede contradictory info)
        """
        raw_hash = hashlib.sha256(raw_text.encode()).hexdigest()
        if self.genome.has_hash(raw_hash):
            return None

        # Enable batch mode to avoid redundant disk writes during store
        self.genome.begin_batch()
        self.entity_registry.begin_batch()
        self.graph.begin_batch()

        try:
            # RNA transcription -> DNA compression (1 API call)
            strand = self.encoder.encode(raw_text, timestamp=timestamp)

            # Check for strand versioning — does this supersede an existing strand?
            self._check_supersede(strand)

            # Store DNA in genome
            self.genome.add(strand)

            # Build neural connections
            self.graph.add_strand(
                strand,
                recent_ids=self._recent_ids,
                entity_registry=self.entity_registry,
                genome_getter=self.genome.get,
            )

            # Bidirectional ego linking
            should_ego_link = (
                strand.modifier in self.EGO_MODIFIERS
                or strand.relation in self.AGENT_ACTION_RELATIONS
            )
            if should_ego_link:
                self.graph.link_to_ego(strand.strand_id, "agent")

            self._recent_ids.append(strand.strand_id)
            return strand
        finally:
            # Single save at end of store operation
            self.genome.end_batch()
            self.entity_registry.end_batch()
            self.graph.end_batch()

    def _check_supersede(self, new_strand: CodebookStrand):
        """
        Strand versioning: if a new strand has the same primary entity and a
        supersedable relation as an existing strand, and the sentiment differs,
        the old one is superseded.

        Only belief-state relations (HESITANT, WENT_QUIET, TRIAL_POSITIVE, etc.)
        are supersedable. Factual/event relations (SENT, SCHEDULED, MENTIONED)
        are historical records and are never superseded.
        """
        if not new_strand.entity_slots:
            return

        # Only supersede belief-state relations
        if new_strand.relation not in self.SUPERSEDABLE_RELATIONS:
            return

        new_primary = new_strand.entity_slots[0][1]
        new_relation = new_strand.relation

        for existing in self.genome.active_strands():
            if not existing.entity_slots:
                continue
            ex_primary = existing.entity_slots[0][1]
            ex_relation = existing.relation

            # Same entity + same relation + different sentiment = updated belief
            if (ex_primary == new_primary
                    and ex_relation == new_relation
                    and existing.sentiment != new_strand.sentiment):
                self.genome.supersede(existing.strand_id, new_strand.strand_id)
                # Weaken old strand's edges
                for _, _, d in list(self.graph.graph.edges(existing.strand_id, data=True)):
                    d["weight"] *= 0.5

    # ── Retrieval Pipeline (Neuroscience) ────────────────────────────────

    def query(self, query_text: str) -> dict:
        """
        Brain-perfect retrieval:
        1. Encode query -> DNA (1 API call)
        2. Spreading activation with all brain mechanisms (local, 0 calls)
        3. LLM reads DNA + traces directly (1 API call)
        Total: 2 API calls. Fixed context cost.
        """
        query_strand = self.encoder.encode(query_text)
        return self.expression.express(query_text, query_strand)

    # ── Memory Consolidation ("Sleep") ───────────────────────────────────

    def consolidate(self) -> dict:
        """
        Brain-like memory consolidation. The brain does this during sleep:
        - Merge related strands (same entity + same relation cluster)
        - Strengthen important edges
        - Create consolidated super-strands

        Call periodically (e.g., after every N store() calls).
        Returns stats about what was consolidated.
        """
        groups: dict[tuple[str, int], list[CodebookStrand]] = defaultdict(list)

        # Group active strands by (primary_entity, relation_cluster)
        for strand in self.genome.active_strands():
            if not strand.entity_slots:
                continue
            primary = strand.entity_slots[0][1]
            # Find which cluster this relation belongs to
            cluster_id = self._get_relation_cluster(strand.relation)
            groups[(primary, cluster_id)].append(strand)

        consolidated = 0
        for (entity, cluster), strands in groups.items():
            if len(strands) < 3:
                continue  # not enough to consolidate

            # Sort by timestamp, keep the most recent
            strands.sort(key=lambda s: s.timestamp, reverse=True)
            keeper = strands[0]

            # Combine traces from all strands in the group
            all_traces = [s.trace for s in strands if s.trace]
            if all_traces:
                combined_trace = " | ".join(all_traces[:3])  # keep top 3 traces
                keeper.trace = combined_trace

            # Boost keeper's confidence (consolidated = more certain)
            keeper.confidence = min(5, keeper.confidence + 1)

            # Supersede older strands
            for old in strands[1:]:
                self.genome.supersede(old.strand_id, keeper.strand_id)
                consolidated += 1

        if consolidated > 0:
            self.genome.save()

        return {"consolidated": consolidated, "groups_found": len(groups), "groups_processed": len(groups), "merged": consolidated}

    def _get_relation_cluster(self, relation_code: int) -> int:
        """Find which semantic cluster a relation belongs to."""
        for cluster_id, codes in self.codebook._RELATION_CLUSTERS.items():
            if relation_code in codes:
                return cluster_id
        return -1  # unclustered

    # ── Intelligent Forgetting ───────────────────────────────────────────

    def forget(self, min_age_seconds: int | None = None, min_activations: int | None = None) -> dict:
        """
        Brain-like forgetting. Prune strands that:
        - Have never been activated in any query
        - Are older than min_age_seconds
        - Are NOT ego-linked (important memories protected)
        - Are NOT the only strand for their primary entity

        Returns stats about what was forgotten.
        """
        if min_age_seconds is None:
            min_age_seconds = self.config.forget_min_age_seconds
        if min_activations is None:
            min_activations = self.config.forget_min_activations

        now = int(time.time())
        ego_linked = set(self.graph.get_ego_linked_strands("agent"))
        forgotten = 0

        # Count strands per entity to avoid forgetting the last reference
        entity_counts: dict[str, int] = defaultdict(int)
        for strand in self.genome.active_strands():
            if strand.entity_slots:
                entity_counts[strand.entity_slots[0][1]] += 1

        for strand in list(self.genome.active_strands()):
            age = now - strand.timestamp
            if age < min_age_seconds:
                continue
            if strand.activation_count > min_activations:
                continue
            if strand.strand_id in ego_linked:
                continue  # protected by ego
            if strand.entity_slots:
                primary = strand.entity_slots[0][1]
                if entity_counts.get(primary, 0) <= 1:
                    continue  # don't forget the only reference to an entity

            # This memory was never useful — forget it
            self.genome.remove(strand.strand_id)
            if self.graph.graph.has_node(strand.strand_id):
                self.graph.graph.remove_node(strand.strand_id)
            forgotten += 1

            if strand.entity_slots:
                entity_counts[strand.entity_slots[0][1]] -= 1

        if forgotten > 0:
            self.graph.save()

        return {"forgotten": forgotten}

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = self.genome.count()
        active = len(self.genome.active_ids())
        superseded = total - active
        return {
            "total_strands": total,
            "active_strands": active,
            "superseded_strands": superseded,
            "graph_nodes": self.graph.node_count(),
            "graph_edges": self.graph.edge_count(),
            "entity_instances": self.entity_registry.count(),
            "entities": self.entity_registry.count(),
            "ego_nodes": self.graph.ego_node_count(),
            "ego_linked_strands": len(self.graph.get_ego_linked_strands("agent")),
            "codebook_size": self.codebook.total_codes(),
        }
