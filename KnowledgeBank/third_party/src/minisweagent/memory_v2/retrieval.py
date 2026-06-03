from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
from difflib import SequenceMatcher

from .schema import MemoryRecord


def score_memory_record(
    *,
    relevance: float,
    confidence: float,
    success_prior: float,
    recency_bonus: float,
    usage_bonus: float,
    conflict_penalty: float,
    staleness_penalty: float,
    ) -> float:
    return (
        relevance
        + confidence
        + success_prior
        + recency_bonus
        + usage_bonus
        - conflict_penalty
        - staleness_penalty
    )


@dataclass(slots=True)
class RankedMemory:
    record: MemoryRecord
    score: float
    relevance: float


def select_top_memories(ranked: list[RankedMemory], limit: int) -> list[MemoryRecord]:
    ranked = sorted(ranked, key=lambda item: item.score, reverse=True)
    return [item.record for item in ranked[:limit]]


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]{3,}", text.lower())}


def compute_text_relevance(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens) / len(query_tokens | text_tokens)
    surface = SequenceMatcher(None, query.lower(), text.lower()).ratio()
    return (overlap * 0.7) + (surface * 0.3)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _age_days(value: str | None, *, now: datetime) -> float | None:
    dt = _parse_timestamp(value)
    if dt is None:
        return None
    return max((now - dt).total_seconds() / 86400.0, 0.0)


def rank_memory_record(record: MemoryRecord, query: str, *, now: str | None = None) -> RankedMemory:
    now_dt = _parse_timestamp(now) or datetime.now(timezone.utc)
    relevance = compute_text_relevance(query, f"{record.query}\n{record.content}")
    verifier_label = record.verifier_label or record.source_status
    if verifier_label == "verified_success":
        success_prior = 0.30
    elif verifier_label == "success":
        success_prior = 0.25
    elif verifier_label == "uncertain":
        success_prior = -0.25
    elif verifier_label in {"verified_fail", "fail"}:
        success_prior = -0.10
    else:
        success_prior = 0.05
    last_touch_age = _age_days(record.last_used_at or record.created_at, now=now_dt)
    if last_touch_age is None:
        recency_bonus = 0.0
    else:
        recency_bonus = max(0.0, 0.25 - min(last_touch_age / 90.0, 0.25))
    usage_bonus = min(math.log1p(record.use_count) / 8.0, 0.2)
    conflict_penalty = min(len(record.conflicts_with) * 0.15, 0.45)
    created_age = _age_days(record.created_at, now=now_dt) or 0.0
    stale_failure = record.source_status != "success" and created_age > 14 and record.use_count == 0
    staleness_penalty = 0.25 if stale_failure else max(0.0, min(created_age / 365.0, 0.2) - usage_bonus / 2.0)
    summary_penalty = 0.12 if record.state.value == "summary" else 0.0
    verifier_penalty = 0.0
    if record.verifier_score is not None and record.verifier_score < 0.55:
        verifier_penalty += 0.45
    if verifier_label == "uncertain":
        verifier_penalty += 0.35
    score = score_memory_record(
        relevance=relevance,
        confidence=record.confidence,
        success_prior=success_prior,
        recency_bonus=recency_bonus,
        usage_bonus=usage_bonus,
        conflict_penalty=conflict_penalty,
        staleness_penalty=staleness_penalty + summary_penalty + verifier_penalty,
    )
    return RankedMemory(record=record, score=score, relevance=relevance)


def rank_memory_records(records: list[MemoryRecord], query: str, *, now: str | None = None) -> list[RankedMemory]:
    return [rank_memory_record(record, query, now=now) for record in records]


def build_memory_block(records: list[MemoryRecord]) -> str:
    if not records:
        return ""
    sections = []
    for index, record in enumerate(records, start=1):
        sections.append(
            "\n".join(
                [
                    f"Memory {index}",
                    f"Type: {record.memory_type.value}",
                    f"Confidence: {record.confidence:.2f}",
                    record.content.strip(),
                ]
            )
        )
    return "\n\n".join(sections)
