from __future__ import annotations

from difflib import SequenceMatcher
from typing import Iterable

from .conflict import may_conflict
from .retrieval import compute_text_relevance
from .schema import MemoryRecord, MemoryState
from .signatures import signature_similarity


def choose_memory_edit_action(new_content: str, existing_candidates: list[str]) -> str:
    if not existing_candidates:
        return "ADD"

    if any(may_conflict(new_content, candidate) for candidate in existing_candidates):
        return "REPLACE"

    best_surface = max(SequenceMatcher(None, new_content, candidate).ratio() for candidate in existing_candidates)
    best_signature = max(signature_similarity(new_content, candidate) for candidate in existing_candidates)
    if best_surface >= 0.95:
        return "DISCARD"
    if best_surface >= 0.80 or best_signature >= 0.5:
        return "MERGE"
    return "ADD"


def choose_best_match(candidate: MemoryRecord, active_records: Iterable[MemoryRecord]) -> MemoryRecord | None:
    ranked = sorted(
        active_records,
        key=lambda record: max(
            compute_text_relevance(candidate.query, record.query),
            compute_text_relevance(candidate.content, record.content),
            signature_similarity(candidate.content, record.content),
        ),
        reverse=True,
    )
    if not ranked:
        return None
    best = ranked[0]
    best_score = max(
        compute_text_relevance(candidate.query, best.query),
        compute_text_relevance(candidate.content, best.content),
        signature_similarity(candidate.content, best.content),
    )
    return best if best_score >= 0.35 or best.dedup_key == candidate.dedup_key else None


def merge_memory_records(primary: MemoryRecord, secondary: MemoryRecord) -> MemoryRecord:
    merged_parts = []
    for part in (primary.content.strip(), secondary.content.strip()):
        if part and part not in merged_parts:
            merged_parts.append(part)
    merged = primary.model_copy(deep=True)
    merged.content = "\n".join(merged_parts)
    merged.confidence = max(primary.confidence, secondary.confidence)
    merged.quality_score = max(primary.quality_score, secondary.quality_score)
    merged.use_count = max(primary.use_count, secondary.use_count)
    merged.last_used_at = primary.last_used_at or secondary.last_used_at
    merged.supersedes = list(dict.fromkeys(primary.supersedes + secondary.supersedes + [secondary.memory_id]))
    merged.conflicts_with = list(dict.fromkeys(primary.conflicts_with + secondary.conflicts_with))
    return merged


def resolve_candidate(
    candidate: MemoryRecord,
    active_records: list[MemoryRecord],
) -> tuple[list[MemoryRecord], list[MemoryRecord], str]:
    best = choose_best_match(candidate, active_records)
    if best is None:
        promoted = candidate.model_copy(update={"state": MemoryState.ACTIVE})
        return active_records + [promoted], [], "ADD"

    if best.dedup_key == candidate.dedup_key and not may_conflict(candidate.content, best.content):
        ratio = SequenceMatcher(None, candidate.content, best.content).ratio()
        action = "DISCARD" if ratio >= 0.92 else "MERGE"
    else:
        action = choose_memory_edit_action(candidate.content, [best.content])
    archived: list[MemoryRecord] = []
    updated_active = [record for record in active_records if record.memory_id != best.memory_id]

    if action == "DISCARD":
        archived.append(candidate.model_copy(update={"state": MemoryState.ARCHIVE}))
        return active_records, archived, action

    if action == "MERGE":
        merged = merge_memory_records(best, candidate)
        merged.state = MemoryState.ACTIVE
        return updated_active + [merged], archived, action

    if action == "REPLACE":
        existing_strength = (best.quality_score, best.confidence, best.source_status == "success")
        candidate_strength = (candidate.quality_score, candidate.confidence, candidate.source_status == "success")
        if candidate_strength >= existing_strength:
            archived.append(best.model_copy(update={"state": MemoryState.ARCHIVE}))
            promoted = candidate.model_copy(
                update={
                    "state": MemoryState.ACTIVE,
                    "conflicts_with": list(dict.fromkeys(candidate.conflicts_with + [best.memory_id])),
                }
            )
            return updated_active + [promoted], archived, action
        archived.append(candidate.model_copy(update={"state": MemoryState.ARCHIVE, "conflicts_with": [best.memory_id]}))
        return active_records, archived, "DISCARD"

    promoted = candidate.model_copy(update={"state": MemoryState.ACTIVE})
    return active_records + [promoted], archived, action


def consolidate_active_records(active_records: list[MemoryRecord]) -> tuple[list[MemoryRecord], list[MemoryRecord]]:
    survivors: list[MemoryRecord] = []
    archived: list[MemoryRecord] = []
    ordered = sorted(
        active_records,
        key=lambda record: (record.quality_score, record.confidence, record.source_status == "success"),
        reverse=True,
    )
    for record in ordered:
        updated_survivors, newly_archived, action = resolve_candidate(record, survivors)
        if action == "ADD":
            survivors = updated_survivors
            continue
        if action in {"MERGE", "REPLACE"}:
            survivors = updated_survivors
            archived.extend(newly_archived)
            continue
        archived.extend(newly_archived)
    return survivors, archived
