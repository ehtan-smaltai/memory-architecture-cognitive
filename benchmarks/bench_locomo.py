#!/usr/bin/env python3
"""
LoCoMo Benchmark Runner — Evaluating cognitive memory on the LoCoMo benchmark.

LoCoMo (Long-Context Conversational Memory) is the standard benchmark for
evaluating AI agent memory systems, from the ACL 2024 paper:
"Evaluating Very Long-Term Conversational Memory of LLM Agents"
(Maharana et al., Snap Research, arXiv:2402.17753)

Dataset: 10 multi-session conversations, ~300 turns each, ~9K tokens avg.
QA types: single-hop, temporal, multi-hop, open-domain, adversarial.
Metrics: token-level F1, LLM-as-Judge (J-score).

Usage:
    # Download dataset first
    python -m benchmarks.bench_locomo --download

    # Run full benchmark (requires ANTHROPIC_API_KEY)
    python -m benchmarks.bench_locomo

    # Run on specific conversations
    python -m benchmarks.bench_locomo --conversations 0,1,2

    # Run with specific question categories only
    python -m benchmarks.bench_locomo --categories 1,2,3

    # Dry run (no API calls, just load and validate)
    python -m benchmarks.bench_locomo --dry-run

    # Run with cost tracking
    python -m benchmarks.bench_locomo --track-cost
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cognitive_memory import Config, MemorySystem


# ─── Constants ────────────────────────────────────────────────────────────────

LOCOMO_DATA_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
LOCOMO_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOCOMO_DATA_FILE = os.path.join(LOCOMO_DATA_DIR, "locomo10.json")

# QA category names (1-indexed in the dataset)
CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}

# Judge model for LLM-as-Judge scoring
JUDGE_MODEL = "claude-sonnet-4-20250514"

# Encoding model for memory system
MEMORY_MODEL = "claude-sonnet-4-20250514"


# ─── Data structures ─────────────────────────────────────────────────────────


@dataclass
class DialogueTurn:
    """A single dialogue turn in a conversation."""

    speaker: str
    dia_id: str
    text: str
    session_num: int
    session_datetime: str


@dataclass
class QAItem:
    """A single QA evaluation item."""

    question: str
    answer: str  # gold answer
    category: int  # 1-5
    evidence: list[str]  # list of dia_ids
    adversarial_answer: str = ""


@dataclass
class Conversation:
    """A full LoCoMo conversation with sessions and QA items."""

    sample_id: str
    speaker_a: str
    speaker_b: str
    turns: list[DialogueTurn]
    qa_items: list[QAItem]


@dataclass
class EvalResult:
    """Result for a single QA evaluation."""

    question: str
    gold_answer: str
    predicted_answer: str
    category: int
    f1: float
    judge_score: float  # 0.0 or 1.0 from LLM-as-Judge
    tokens_used: int
    api_calls: int
    latency_ms: float


@dataclass
class BenchmarkReport:
    """Full benchmark results."""

    results: list[EvalResult] = field(default_factory=list)
    total_store_time_s: float = 0.0
    total_store_api_calls: int = 0
    total_strands_stored: int = 0
    total_query_api_calls: int = 0
    total_judge_api_calls: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0


# ─── Dataset loading ─────────────────────────────────────────────────────────


def download_locomo():
    """Download the LoCoMo dataset from GitHub."""
    import urllib.request

    os.makedirs(LOCOMO_DATA_DIR, exist_ok=True)

    if os.path.exists(LOCOMO_DATA_FILE):
        print(f"  Dataset already exists at {LOCOMO_DATA_FILE}")
        return

    print(f"  Downloading LoCoMo dataset from {LOCOMO_DATA_URL}...")
    urllib.request.urlretrieve(LOCOMO_DATA_URL, LOCOMO_DATA_FILE)
    print(f"  Saved to {LOCOMO_DATA_FILE}")


def load_locomo(path: str = LOCOMO_DATA_FILE) -> list[Conversation]:
    """Load and parse the LoCoMo dataset."""
    with open(path, "r") as f:
        raw_data = json.load(f)

    conversations = []
    for record in raw_data:
        conv_data = record["conversation"]
        speaker_a = conv_data.get("speaker_a", "Speaker A")
        speaker_b = conv_data.get("speaker_b", "Speaker B")

        # Parse all sessions
        turns = []
        session_num = 1
        while True:
            session_key = f"session_{session_num}"
            datetime_key = f"session_{session_num}_date_time"

            if session_key not in conv_data:
                break

            session_datetime = conv_data.get(datetime_key, f"Session {session_num}")
            session_turns = conv_data[session_key]

            for turn_data in session_turns:
                turns.append(
                    DialogueTurn(
                        speaker=turn_data["speaker"],
                        dia_id=turn_data.get("dia_id", ""),
                        text=turn_data.get("text", turn_data.get("clean_text", "")),
                        session_num=session_num,
                        session_datetime=session_datetime,
                    )
                )

            session_num += 1

        # Parse QA items
        qa_items = []
        for qa_data in record.get("qa", []):
            category = qa_data["category"]
            # Category 5 (adversarial) items have no 'answer' field —
            # they only have 'adversarial_answer' (the distractor).
            # The correct behavior is to NOT produce the adversarial answer.
            if category == 5:
                # For adversarial Qs, the gold answer is essentially
                # "unanswerable" or "not mentioned". We use a sentinel
                # so the judge can evaluate whether the system was fooled.
                gold_answer = qa_data.get("answer", "This information is not mentioned in the conversation.")
                adversarial = qa_data.get("adversarial_answer", "")
            else:
                gold_answer = qa_data["answer"]
                adversarial = ""

            qa_items.append(
                QAItem(
                    question=qa_data["question"],
                    answer=gold_answer,
                    category=category,
                    evidence=qa_data.get("evidence", []),
                    adversarial_answer=adversarial,
                )
            )

        conversations.append(
            Conversation(
                sample_id=record.get("sample_id", f"conv_{len(conversations)}"),
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                turns=turns,
                qa_items=qa_items,
            )
        )

    return conversations


# ─── F1 Scoring (matches LoCoMo's official implementation) ───────────────────

# Porter stemmer — lightweight implementation to avoid nltk dependency
# This matches the LoCoMo paper's use of PorterStemmer for token-level F1.

try:
    from nltk.stem import PorterStemmer as _PorterStemmer

    _stemmer = _PorterStemmer()
except ImportError:
    # Fallback: no stemming (slightly lower F1, but avoids dependency)
    class _FallbackStemmer:
        def stem(self, word: str) -> str:
            return word

    _stemmer = _FallbackStemmer()


def normalize_answer(s: str) -> str:
    """Normalize answer text for F1 computation. Matches LoCoMo's implementation."""
    s = s.replace(",", "")

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score between prediction and ground truth.
    Matches LoCoMo's official evaluation: normalize, stem, then compute F1.
    """
    pred_tokens = [_stemmer.stem(w) for w in normalize_answer(prediction).split()]
    gold_tokens = [_stemmer.stem(w) for w in normalize_answer(ground_truth).split()]

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# ─── LLM-as-Judge scoring ───────────────────────────────────────────────────

JUDGE_PROMPT = """You are an impartial judge evaluating the quality of an AI assistant's answer to a question about a long conversation.

You will be given:
- A question about the conversation
- The ground truth (correct) answer
- The AI assistant's predicted answer

Your task is to determine if the predicted answer is CORRECT — meaning it captures the essential factual content of the ground truth answer.

Rules:
- The predicted answer does NOT need to be word-for-word identical
- It MUST contain the key facts from the ground truth
- Partial answers that capture the main point count as correct
- Answers with hallucinated facts that contradict the ground truth are INCORRECT
- "I don't know" or refusals when the ground truth has a real answer are INCORRECT

Respond with ONLY one word: CORRECT or INCORRECT"""


ADVERSARIAL_JUDGE_PROMPT = """You are an impartial judge evaluating whether an AI assistant was fooled by an adversarial question.

The question is designed to trick the AI into producing a WRONG answer (the "adversarial answer").
The conversation does NOT contain information to answer this question correctly.

A CORRECT response is one that:
- Says the information is not available, not mentioned, or unknown
- Refuses to answer or expresses uncertainty
- Does NOT produce the adversarial (wrong) answer

An INCORRECT response is one that:
- Produces the adversarial answer or something similar to it
- Makes up a confident-sounding but wrong answer
- Hallucinate facts not present in the conversation

Respond with ONLY one word: CORRECT or INCORRECT"""


def judge_answer(
    client: anthropic.Anthropic,
    question: str,
    gold_answer: str,
    predicted_answer: str,
    model: str = JUDGE_MODEL,
    is_adversarial: bool = False,
    adversarial_answer: str = "",
) -> float:
    """
    Use LLM-as-Judge to score a predicted answer against the gold answer.
    Returns 1.0 for CORRECT, 0.0 for INCORRECT.

    For adversarial questions, checks that the system was NOT fooled
    (i.e., did not produce the adversarial distractor answer).
    """
    if is_adversarial:
        system_prompt = ADVERSARIAL_JUDGE_PROMPT
        user_msg = f"""Question: {question}

Adversarial (wrong) answer that would indicate the AI was tricked: {adversarial_answer}

AI assistant's response: {predicted_answer}"""
    else:
        system_prompt = JUDGE_PROMPT
        user_msg = f"""Question: {question}

Ground truth answer: {gold_answer}

Predicted answer: {predicted_answer}"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=10,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        verdict = response.content[0].text.strip().upper()
        return 1.0 if "CORRECT" in verdict else 0.0
    except anthropic.APIError as e:
        print(f"    [WARN] Judge API error: {e}")
        return 0.0


# ─── Cost estimation ─────────────────────────────────────────────────────────

# Approximate token costs for Claude Sonnet (per 1M tokens)
COST_PER_1M_INPUT = 3.00  # $3/MTok input
COST_PER_1M_OUTPUT = 15.00  # $15/MTok output


def estimate_cost(
    input_tokens: int, output_tokens: int, model: str = MEMORY_MODEL
) -> float:
    """Estimate cost in USD for a given number of tokens."""
    return (input_tokens / 1_000_000) * COST_PER_1M_INPUT + (
        output_tokens / 1_000_000
    ) * COST_PER_1M_OUTPUT


# ─── Ingestion phase ─────────────────────────────────────────────────────────


def ingest_conversation(
    mem: MemorySystem,
    conv: Conversation,
    verbose: bool = False,
) -> dict:
    """
    Store all turns from a conversation into the memory system.

    Each turn is stored as a separate memory, preserving the speaker,
    session context, and temporal ordering.

    Returns ingestion stats.
    """
    stats = {"turns": 0, "strands_stored": 0, "skipped": 0, "errors": 0}

    # Assign timestamps: base + session offset + turn offset
    # This preserves temporal ordering across sessions
    base_ts = 1_600_000_000  # Sept 2020 as base
    session_gap = 86400 * 7  # 1 week between sessions
    turn_gap = 300  # 5 minutes between turns

    current_session = 0
    turn_in_session = 0

    for i, turn in enumerate(conv.turns):
        if turn.session_num != current_session:
            current_session = turn.session_num
            turn_in_session = 0

        timestamp = base_ts + (turn.session_num * session_gap) + (turn_in_session * turn_gap)
        turn_in_session += 1

        # Format the turn with speaker context and session info
        formatted = (
            f"[Session {turn.session_num}, {turn.session_datetime}] "
            f"{turn.speaker}: {turn.text}"
        )

        try:
            strand = mem.store(formatted, timestamp=timestamp)
            stats["turns"] += 1
            if strand is not None:
                stats["strands_stored"] += 1
            else:
                stats["skipped"] += 1  # duplicate

            if verbose and (i + 1) % 50 == 0:
                print(f"    Stored {i + 1}/{len(conv.turns)} turns...")

        except Exception as e:
            stats["errors"] += 1
            if verbose:
                print(f"    [ERROR] Turn {i}: {e}")

    # Run consolidation after ingesting all turns
    consolidate_stats = mem.consolidate()
    stats["consolidated"] = consolidate_stats.get("consolidated", 0)

    return stats


# ─── Query phase ──────────────────────────────────────────────────────────────


def evaluate_qa(
    mem: MemorySystem,
    qa_item: QAItem,
    judge_client: anthropic.Anthropic,
    skip_judge: bool = False,
    verbose: bool = False,
) -> EvalResult:
    """
    Evaluate a single QA item against the memory system.

    1. Query the memory system with the question
    2. Compute F1 against gold answer
    3. Run LLM-as-Judge for J-score
    """
    start = time.perf_counter()

    # Query the memory system
    result = mem.query(qa_item.question)
    predicted = result.get("answer", "")

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Compute F1
    # For multi-hop (cat 3) answers with semicolons, the official LoCoMo eval
    # only uses the part before the first semicolon.
    gold_for_f1 = qa_item.answer
    if qa_item.category == 3 and ";" in qa_item.answer:
        gold_for_f1 = qa_item.answer.split(";")[0].strip()
    f1 = compute_f1(predicted, gold_for_f1)

    # LLM-as-Judge
    judge_score = 0.0
    if not skip_judge:
        is_adversarial = qa_item.category == 5
        judge_score = judge_answer(
            judge_client,
            qa_item.question,
            qa_item.answer,
            predicted,
            is_adversarial=is_adversarial,
            adversarial_answer=qa_item.adversarial_answer,
        )

    tokens_used = result.get("tokens_used", 0)
    # 2 API calls for query (encode + reason), 1 for judge
    api_calls = 2 + (0 if skip_judge else 1)

    if verbose:
        cat_name = CATEGORY_NAMES.get(qa_item.category, "unknown")
        status = "PASS" if judge_score > 0.5 else "FAIL"
        print(f"    [{cat_name:>11}] F1={f1:.3f} J={judge_score:.0f} {status}")
        if verbose and f1 < 0.3:
            print(f"      Q: {qa_item.question[:80]}...")
            print(f"      Gold: {qa_item.answer[:80]}...")
            print(f"      Pred: {predicted[:80]}...")

    return EvalResult(
        question=qa_item.question,
        gold_answer=qa_item.answer,
        predicted_answer=predicted,
        category=qa_item.category,
        f1=f1,
        judge_score=judge_score,
        tokens_used=tokens_used,
        api_calls=api_calls,
        latency_ms=elapsed_ms,
    )


# ─── Full benchmark runner ───────────────────────────────────────────────────


def run_benchmark(
    conversations: list[Conversation],
    categories: list[int] | None = None,
    skip_judge: bool = False,
    verbose: bool = False,
    track_cost: bool = False,
    model: str = MEMORY_MODEL,
    token_budget: int = 500,
) -> BenchmarkReport:
    """
    Run the full LoCoMo benchmark.

    For each conversation:
    1. Create a fresh MemorySystem
    2. Ingest all dialogue turns
    3. Evaluate all QA items
    4. Compute metrics
    """
    report = BenchmarkReport()
    judge_client = anthropic.Anthropic()

    for conv_idx, conv in enumerate(conversations):
        print(f"\n{'='*70}")
        print(f"  Conversation {conv_idx + 1}/{len(conversations)}: {conv.sample_id}")
        print(f"  Speakers: {conv.speaker_a} & {conv.speaker_b}")
        print(f"  Turns: {len(conv.turns)}, QA items: {len(conv.qa_items)}")
        print(f"{'='*70}")

        # Create fresh memory system in temp directory
        tmpdir = tempfile.mkdtemp(prefix=f"locomo_bench_{conv_idx}_")
        config = Config(
            genome_path=os.path.join(tmpdir, "genome.json"),
            graph_path=os.path.join(tmpdir, "graph.json"),
            entities_path=os.path.join(tmpdir, "entities.json"),
            model=model,
            token_budget=token_budget,
        )
        mem = MemorySystem(config=config)

        # Phase 1: Ingest
        print(f"\n  Phase 1: Ingesting {len(conv.turns)} turns...")
        ingest_start = time.perf_counter()
        ingest_stats = ingest_conversation(mem, conv, verbose=verbose)
        ingest_time = time.perf_counter() - ingest_start

        report.total_store_time_s += ingest_time
        report.total_strands_stored += ingest_stats["strands_stored"]
        # 1 API call per stored strand (encoding)
        report.total_store_api_calls += ingest_stats["strands_stored"]

        mem_stats = mem.stats()
        print(f"  Ingested: {ingest_stats['strands_stored']} strands in {ingest_time:.1f}s")
        print(f"  Consolidated: {ingest_stats.get('consolidated', 0)} strands")
        print(f"  Memory: {mem_stats['active_strands']} active, "
              f"{mem_stats['graph_edges']} edges, "
              f"{mem_stats['entity_instances']} entities")

        # Phase 2: Evaluate QA
        qa_items = conv.qa_items
        if categories:
            qa_items = [q for q in qa_items if q.category in categories]

        print(f"\n  Phase 2: Evaluating {len(qa_items)} QA items...")
        for qi_idx, qa_item in enumerate(qa_items):
            result = evaluate_qa(
                mem, qa_item, judge_client,
                skip_judge=skip_judge, verbose=verbose,
            )
            report.results.append(result)
            report.total_tokens_used += result.tokens_used
            report.total_query_api_calls += 2  # encode + reason
            if not skip_judge:
                report.total_judge_api_calls += 1

            if not verbose and (qi_idx + 1) % 20 == 0:
                print(f"    Evaluated {qi_idx + 1}/{len(qa_items)}...")

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    return report


# ─── Reporting ───────────────────────────────────────────────────────────────


def print_report(report: BenchmarkReport, skip_judge: bool = False):
    """Print a formatted benchmark report."""
    if not report.results:
        print("\n  No results to report.")
        return

    print(f"\n{'='*70}")
    print("  LOCOMO BENCHMARK RESULTS — Cognitive Memory Architecture")
    print(f"{'='*70}")

    # Overall metrics
    all_f1 = [r.f1 for r in report.results]
    all_j = [r.judge_score for r in report.results]
    avg_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0
    avg_j = sum(all_j) / len(all_j) if all_j else 0

    print(f"\n  OVERALL ({len(report.results)} questions)")
    print(f"  {'─'*50}")
    print(f"  F1 Score:        {avg_f1:.4f}  ({avg_f1*100:.1f}%)")
    if not skip_judge:
        print(f"  J-Score:         {avg_j:.4f}  ({avg_j*100:.1f}%)")
    print(f"  Avg Latency:     {sum(r.latency_ms for r in report.results) / len(report.results):.0f} ms")
    print(f"  Avg Tokens/Query:{sum(r.tokens_used for r in report.results) / len(report.results):.0f}")

    # Per-category breakdown
    print(f"\n  PER-CATEGORY BREAKDOWN")
    print(f"  {'─'*65}")
    header = f"  {'Category':>14} | {'Count':>5} | {'F1':>8}"
    if not skip_judge:
        header += f" | {'J-Score':>8}"
    header += f" | {'Avg Tokens':>10} | {'Avg ms':>8}"
    print(header)
    print(f"  {'─'*65}")

    for cat in sorted(CATEGORY_NAMES.keys()):
        cat_results = [r for r in report.results if r.category == cat]
        if not cat_results:
            continue

        cat_f1 = sum(r.f1 for r in cat_results) / len(cat_results)
        cat_j = sum(r.judge_score for r in cat_results) / len(cat_results)
        cat_tokens = sum(r.tokens_used for r in cat_results) / len(cat_results)
        cat_latency = sum(r.latency_ms for r in cat_results) / len(cat_results)
        cat_name = CATEGORY_NAMES[cat]

        line = f"  {cat_name:>14} | {len(cat_results):>5} | {cat_f1:>7.3f}"
        if not skip_judge:
            line += f" | {cat_j:>7.3f}"
        line += f" | {cat_tokens:>10.0f} | {cat_latency:>7.0f}"
        print(line)

    # Comparison with published scores
    print(f"\n  COMPARISON WITH PUBLISHED SCORES")
    print(f"  {'─'*55}")
    if not skip_judge:
        print(f"  {'System':>24} | {'J-Score':>10} | {'Notes':>16}")
        print(f"  {'─'*55}")
        print(f"  {'>>> OURS <<<':>24} | {avg_j*100:>9.1f}% | cognitive memory")
        print(f"  {'─'*55}")
        print(f"  {'EverMemOS':>24} | {'~92.3%':>10} | GPT-4.1-mini")
        print(f"  {'MemMachine v0.2':>24} | {'~84.9%':>10} | Dec 2025")
        print(f"  {'Zep (Graphiti)':>24} | {'~75.1%':>10} | graph-based")
        print(f"  {'Full-context':>24} | {'~72.9%':>10} | send all turns")
        print(f"  {'Mem0g (graph)':>24} | {'~68.4%':>10} | graph memory")
        print(f"  {'Mem0':>24} | {'~66.9%':>10} | base version")
        print(f"  {'OpenAI Memory':>24} | {'~52.9%':>10} | per Mem0 paper")
    else:
        print(f"  {'System':>24} | {'F1 (weighted)':>14}")
        print(f"  {'─'*55}")
        print(f"  {'>>> OURS <<<':>24} | {avg_f1*100:>13.1f}%")
        print(f"  {'─'*55}")
        print(f"  {'SYNAPSE':>24} | {'~40.5%':>14}")
        print(f"  {'A-Mem':>24} | {'~33.3%':>14}")

    # Token efficiency
    print(f"\n  TOKEN EFFICIENCY")
    print(f"  {'─'*50}")
    avg_tokens = sum(r.tokens_used for r in report.results) / len(report.results)
    print(f"  Avg tokens/query:     {avg_tokens:>8.0f}")
    print(f"  Full-context equiv:   ~26,000 tokens")
    print(f"  Token savings:        {(1 - avg_tokens/26000)*100:>7.1f}%")
    if avg_f1 > 0:
        # Cost per query (rough estimate: 2 API calls, ~500 tokens each)
        est_cost = estimate_cost(int(avg_tokens + 500), 400)
        print(f"  Est. cost/query:      ${est_cost:.4f}")
        f1_per_dollar = avg_f1 / est_cost if est_cost > 0 else 0
        print(f"  F1/$:                 {f1_per_dollar:.1f}")

    # Infrastructure stats
    print(f"\n  INFRASTRUCTURE")
    print(f"  {'─'*50}")
    print(f"  Total strands stored: {report.total_strands_stored:>8}")
    print(f"  Total store time:     {report.total_store_time_s:>7.1f}s")
    print(f"  Store API calls:      {report.total_store_api_calls:>8}")
    print(f"  Query API calls:      {report.total_query_api_calls:>8}")
    if not skip_judge:
        print(f"  Judge API calls:      {report.total_judge_api_calls:>8}")
    total_api = (
        report.total_store_api_calls
        + report.total_query_api_calls
        + report.total_judge_api_calls
    )
    print(f"  Total API calls:      {total_api:>8}")
    print(f"{'='*70}\n")


def save_results(report: BenchmarkReport, path: str):
    """Save detailed results to JSON for analysis."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_questions": len(report.results),
            "avg_f1": sum(r.f1 for r in report.results) / len(report.results) if report.results else 0,
            "avg_j_score": sum(r.judge_score for r in report.results) / len(report.results) if report.results else 0,
            "total_strands": report.total_strands_stored,
            "total_api_calls": (
                report.total_store_api_calls
                + report.total_query_api_calls
                + report.total_judge_api_calls
            ),
        },
        "per_category": {},
        "results": [],
    }

    for cat in sorted(CATEGORY_NAMES.keys()):
        cat_results = [r for r in report.results if r.category == cat]
        if cat_results:
            data["per_category"][CATEGORY_NAMES[cat]] = {
                "count": len(cat_results),
                "avg_f1": sum(r.f1 for r in cat_results) / len(cat_results),
                "avg_j_score": sum(r.judge_score for r in cat_results) / len(cat_results),
                "avg_tokens": sum(r.tokens_used for r in cat_results) / len(cat_results),
            }

    for r in report.results:
        data["results"].append({
            "question": r.question,
            "gold_answer": r.gold_answer,
            "predicted_answer": r.predicted_answer,
            "category": CATEGORY_NAMES.get(r.category, "unknown"),
            "f1": round(r.f1, 4),
            "judge_score": r.judge_score,
            "tokens_used": r.tokens_used,
            "latency_ms": round(r.latency_ms, 1),
        })

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to {path}")


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="LoCoMo benchmark for cognitive memory architecture"
    )
    parser.add_argument(
        "--download", action="store_true", help="Download the LoCoMo dataset"
    )
    parser.add_argument(
        "--conversations",
        type=str,
        default=None,
        help="Comma-separated conversation indices (0-indexed), e.g. '0,1,2'",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated QA categories (1-5), e.g. '1,2,3'",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-Judge scoring (faster, F1 only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate data without running evaluation",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output per question"
    )
    parser.add_argument(
        "--track-cost", action="store_true", help="Track and report estimated costs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed results JSON",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MEMORY_MODEL,
        help=f"Model for memory system (default: {MEMORY_MODEL})",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=500,
        help="Token budget per query (default: 500)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Max questions per conversation (for quick testing)",
    )

    args = parser.parse_args()

    # Download
    if args.download:
        download_locomo()
        if not any([args.conversations, args.categories, args.dry_run]):
            return

    # Check dataset exists
    if not os.path.exists(LOCOMO_DATA_FILE):
        print(f"  Dataset not found at {LOCOMO_DATA_FILE}")
        print(f"  Run: python -m benchmarks.bench_locomo --download")
        sys.exit(1)

    # Load
    print("  Loading LoCoMo dataset...")
    conversations = load_locomo()
    print(f"  Loaded {len(conversations)} conversations")

    # Filter conversations
    if args.conversations:
        indices = [int(i.strip()) for i in args.conversations.split(",")]
        conversations = [conversations[i] for i in indices if i < len(conversations)]
        print(f"  Selected {len(conversations)} conversations: {indices}")

    # Print dataset summary
    total_turns = sum(len(c.turns) for c in conversations)
    total_qa = sum(len(c.qa_items) for c in conversations)
    print(f"  Total turns: {total_turns}, Total QA items: {total_qa}")

    for cat in sorted(CATEGORY_NAMES.keys()):
        count = sum(1 for c in conversations for q in c.qa_items if q.category == cat)
        print(f"    Category {cat} ({CATEGORY_NAMES[cat]}): {count} questions")

    if args.dry_run:
        print("\n  Dry run complete. Dataset is valid.")
        return

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n  ERROR: ANTHROPIC_API_KEY not set. Required for benchmark evaluation.")
        sys.exit(1)

    # Filter categories
    categories = None
    if args.categories:
        categories = [int(c.strip()) for c in args.categories.split(",")]
        print(f"  Filtering to categories: {categories}")

    # Limit questions if requested
    if args.max_questions:
        for conv in conversations:
            conv.qa_items = conv.qa_items[: args.max_questions]
        print(f"  Limited to {args.max_questions} questions per conversation")

    # Run benchmark
    print(f"\n  Starting LoCoMo benchmark...")
    print(f"  Model: {args.model}")
    print(f"  Token budget: {args.token_budget}")
    print(f"  Judge: {'ENABLED' if not args.skip_judge else 'DISABLED'}")

    report = run_benchmark(
        conversations,
        categories=categories,
        skip_judge=args.skip_judge,
        verbose=args.verbose,
        track_cost=args.track_cost,
        model=args.model,
        token_budget=args.token_budget,
    )

    # Report
    print_report(report, skip_judge=args.skip_judge)

    # Save results
    output_path = args.output or os.path.join(
        LOCOMO_DATA_DIR, f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_results(report, output_path)


if __name__ == "__main__":
    main()
