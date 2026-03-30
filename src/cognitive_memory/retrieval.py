"""
Hybrid Retrieval — BM25 keyword search + future embedding support.

Provides retrieval signals beyond codebook similarity to fix the
fundamental information loss problem in code-only retrieval.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from .codebook import CodebookStrand


# ─── Tokenizer ──────────────────────────────────────────────────────────────

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
    "has", "had", "have", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "of", "in", "to", "for",
    "on", "at", "by", "with", "from", "about", "as", "and", "or", "but",
    "not", "no", "if", "it", "its", "this", "that", "what", "when",
    "where", "who", "how", "which", "their", "they", "them", "he", "she",
    "his", "her", "we", "our", "you", "your", "i", "me", "my", "so",
    "just", "also", "very", "really", "up", "out", "all", "some", "any",
    "been", "being", "than", "then", "there", "here", "into", "over",
    "such", "only", "more", "most", "other", "each", "both", "few",
    "session",  # noisy in LoCoMo formatted turns
})

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase tokenize, remove stop words."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP_WORDS]


# ─── BM25 Index ─────────────────────────────────────────────────────────────

class BM25Index:
    """
    Lightweight BM25 (Okapi BM25) index over strand raw text + traces.

    No external dependencies. Rebuilt in-memory from genome on init.
    Parameters tuned for conversational text (short documents).
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_tokens: dict[str, list[str]] = {}   # strand_id -> tokens
        self._doc_len: dict[str, int] = {}             # strand_id -> token count
        self._df: dict[str, int] = defaultdict(int)    # term -> doc frequency
        self._tf: dict[str, dict[str, int]] = {}       # strand_id -> {term: count}
        self._avg_dl: float = 0.0
        self._n_docs: int = 0

    def build(self, strands: list[CodebookStrand]) -> None:
        """Build index from active strands."""
        self._doc_tokens.clear()
        self._doc_len.clear()
        self._df.clear()
        self._tf.clear()

        for strand in strands:
            if strand.superseded_by is not None:
                continue
            # Index both raw_text and trace for maximum coverage
            text = f"{strand.raw_text} {strand.trace}"
            tokens = tokenize(text)
            sid = strand.strand_id
            self._doc_tokens[sid] = tokens
            self._doc_len[sid] = len(tokens)

            tf: dict[str, int] = defaultdict(int)
            seen_terms: set[str] = set()
            for token in tokens:
                tf[token] += 1
                if token not in seen_terms:
                    self._df[token] += 1
                    seen_terms.add(token)
            self._tf[sid] = dict(tf)

        self._n_docs = len(self._doc_tokens)
        total_len = sum(self._doc_len.values())
        self._avg_dl = total_len / self._n_docs if self._n_docs > 0 else 1.0

    def add_strand(self, strand: CodebookStrand) -> None:
        """Incrementally add a single strand to the index."""
        text = f"{strand.raw_text} {strand.trace}"
        tokens = tokenize(text)
        sid = strand.strand_id

        self._doc_tokens[sid] = tokens
        self._doc_len[sid] = len(tokens)

        tf: dict[str, int] = defaultdict(int)
        seen_terms: set[str] = set()
        for token in tokens:
            tf[token] += 1
            if token not in seen_terms:
                self._df[token] += 1
                seen_terms.add(token)
        self._tf[sid] = dict(tf)

        # Update corpus stats
        self._n_docs += 1
        total_len = sum(self._doc_len.values())
        self._avg_dl = total_len / self._n_docs if self._n_docs > 0 else 1.0

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """
        Search for strands matching the query.
        Returns list of (strand_id, bm25_score) sorted descending.
        """
        query_tokens = tokenize(query)
        if not query_tokens or self._n_docs == 0:
            return []

        scores: dict[str, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self._df:
                continue
            df = self._df[term]
            # IDF with smoothing
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

            for sid, tf_dict in self._tf.items():
                if term not in tf_dict:
                    continue
                tf = tf_dict[term]
                dl = self._doc_len[sid]
                # BM25 score
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                scores[sid] += idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ─── Reciprocal Rank Fusion ────────────────────────────────────────────────

def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists using RRF (Cormack et al., 2009).

    Each input is a list of (strand_id, score) sorted descending.
    Returns merged list of (strand_id, rrf_score) sorted descending.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (sid, _score) in enumerate(ranked):
            scores[sid] += 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
