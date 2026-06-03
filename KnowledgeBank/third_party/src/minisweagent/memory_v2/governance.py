from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher

from .conflict import may_conflict
from .retrieval import compute_text_relevance
from .schema import MemoryRecord, MemoryState
from .signatures import signature_similarity


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _age_days(value: str | None, *, now: datetime) -> float:
    dt = _parse_timestamp(value)
    if dt is None:
        return 0.0
    return max((now - dt).total_seconds() / 86400.0, 0.0)


def _retention_score(record: MemoryRecord, *, now: datetime) -> float:
    age = _age_days(record.last_used_at or record.created_at, now=now)
    success_bonus = 0.5 if record.source_status == "success" else -0.3
    usage_bonus = min(record.use_count * 0.1, 0.5)
    recency_bonus = max(0.0, 0.25 - min(age / 90.0, 0.25))
    return record.quality_score + record.confidence + success_bonus + usage_bonus + recency_bonus


def _similarity(left: MemoryRecord, right: MemoryRecord) -> float:
    query_sim = compute_text_relevance(left.query, right.query)
    content_sim = compute_text_relevance(left.content, right.content)
    signature_sim = signature_similarity(left.content, right.content)
    surface = SequenceMatcher(None, left.content.lower(), right.content.lower()).ratio()
    return max(query_sim, content_sim, signature_sim, surface * 0.8)


def _make_summary_record(records: list[MemoryRecord], *, now: str) -> MemoryRecord:
    ordered = sorted(records, key=lambda record: (record.quality_score, record.confidence, record.use_count), reverse=True)
    primary = ordered[0]
    unique_lines: list[str] = []
    for record in ordered:
        for line in [part.strip("-* ") for part in record.content.splitlines() if part.strip()]:
            if line and line not in unique_lines:
                unique_lines.append(line)
    summary_lines = unique_lines[:4]
    content = "Summary:\n" + "\n".join(f"- {line}" for line in summary_lines)
    return primary.model_copy(
        update={
            "memory_id": f"summary-{primary.memory_id}",
            "state": MemoryState.SUMMARY,
            "content": content,
            "query": primary.query,
            "dedup_key": f"summary:{primary.dedup_key}",
            "supersedes": list(dict.fromkeys(primary.supersedes + [record.memory_id for record in ordered])),
            "created_at": now,
            "last_used_at": None,
            "use_count": 0,
            "confidence": max(record.confidence for record in ordered),
            "quality_score": max(record.quality_score for record in ordered),
        }
    )


def _stronger_record(record: MemoryRecord, *, now: datetime) -> tuple[float, float, float, int]:
    success_bonus = 1.0 if record.source_status == "success" else 0.0
    return (
        success_bonus,
        _retention_score(record, now=now),
        record.quality_score,
        record.use_count,
    )


def _resolve_conflicts(
    records: list[MemoryRecord],
    *,
    now_dt: datetime,
    similarity_threshold: float,
) -> tuple[list[MemoryRecord], list[MemoryRecord]]:
    archived: list[MemoryRecord] = []
    grouped: dict[str, list[MemoryRecord]] = defaultdict(list)
    for record in records:
        grouped[record.memory_type.value].append(record)

    survivors: list[MemoryRecord] = []
    for candidates in grouped.values():
        ordered = sorted(candidates, key=lambda record: _stronger_record(record, now=now_dt), reverse=True)
        for record in ordered:
            matched_index: int | None = None
            matched_survivor: MemoryRecord | None = None
            for index, survivor in enumerate(survivors):
                if survivor.memory_type != record.memory_type:
                    continue
                close_match = signature_similarity(survivor.content, record.content) >= max(0.25, similarity_threshold - 0.1)
                if close_match and may_conflict(survivor.content, record.content):
                    matched_index = index
                    matched_survivor = survivor
                    break
            if matched_survivor is None or matched_index is None:
                survivors.append(record)
                continue

            winner, loser = sorted(
                [matched_survivor, record],
                key=lambda item: _stronger_record(item, now=now_dt),
                reverse=True,
            )
            archived.append(
                loser.model_copy(
                    update={
                        "state": MemoryState.ARCHIVE,
                        "conflicts_with": list(dict.fromkeys(loser.conflicts_with + [winner.memory_id])),
                    }
                )
            )
            survivors[matched_index] = winner.model_copy(
                update={"supersedes": list(dict.fromkeys(winner.supersedes + [loser.memory_id]))}
            )
    return survivors, archived


def run_budgeted_governance(
    *,
    active_records: list[MemoryRecord],
    summary_records: list[MemoryRecord],
    now: str,
    max_active_records: int,
    similarity_threshold: float,
    cluster_min_size: int,
    retire_failure_days: int,
) -> dict[str, list[MemoryRecord]]:
    now_dt = _parse_timestamp(now) or datetime.now(timezone.utc)

    survivors: list[MemoryRecord] = []
    archived: list[MemoryRecord] = []
    for record in active_records:
        is_stale_failed = (
            record.source_status != "success"
            and record.use_count == 0
            and _age_days(record.created_at, now=now_dt) >= retire_failure_days
        )
        if is_stale_failed:
            archived.append(record.model_copy(update={"state": MemoryState.ARCHIVE}))
            continue
        survivors.append(record)

    survivors, conflicted_archived = _resolve_conflicts(
        survivors,
        now_dt=now_dt,
        similarity_threshold=similarity_threshold,
    )
    archived.extend(conflicted_archived)

    clustered_ids: set[str] = set()
    new_summaries = list(summary_records)
    similarity_groups: dict[str, list[MemoryRecord]] = defaultdict(list)
    for record in survivors:
        if record.source_status == "success":
            similarity_groups[record.memory_type.value].append(record)

    for _, candidates in similarity_groups.items():
        candidates = sorted(candidates, key=lambda record: _retention_score(record, now=now_dt), reverse=True)
        for record in candidates:
            if record.memory_id in clustered_ids:
                continue
            cluster = [record]
            for other in candidates:
                if other.memory_id == record.memory_id or other.memory_id in clustered_ids:
                    continue
                if _similarity(record, other) >= similarity_threshold:
                    cluster.append(other)
            distinct_task_ids = {member.task_id for member in cluster}
            if len(cluster) >= cluster_min_size and len(distinct_task_ids) >= 2:
                for member in cluster:
                    clustered_ids.add(member.memory_id)
                new_summaries.append(_make_summary_record(cluster, now=now))
                archived.extend(member.model_copy(update={"state": MemoryState.ARCHIVE}) for member in cluster)

    survivors = [record for record in survivors if record.memory_id not in clustered_ids]
    ranked_survivors = sorted(survivors, key=lambda record: _retention_score(record, now=now_dt), reverse=True)
    kept_active = ranked_survivors[:max_active_records]
    overflow = ranked_survivors[max_active_records:]
    archived.extend(record.model_copy(update={"state": MemoryState.ARCHIVE}) for record in overflow)

    return {
        "active": kept_active,
        "summary": new_summaries,
        "archived": archived,
    }
