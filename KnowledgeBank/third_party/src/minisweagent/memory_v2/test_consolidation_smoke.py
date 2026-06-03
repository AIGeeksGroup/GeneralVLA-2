from minisweagent.memory_v2.consolidation import choose_memory_edit_action, resolve_candidate
from minisweagent.memory_v2.schema import MemoryRecord, MemoryState, MemoryType


def test_duplicate_fact_prefers_merge_or_discard():
    action = choose_memory_edit_action(
        new_content="Always inspect the failing regression test before editing production code.",
        existing_candidates=[
            "Inspect the existing failing regression tests before editing production logic."
        ],
    )
    assert action in {"MERGE", "DISCARD", "REPLACE"}


def test_resolve_candidate_archives_weaker_conflict():
    active = [
        MemoryRecord(
            memory_id="old",
            task_id="t1",
            query="fix failing test",
            content="Do not patch before reproducing the failing test.",
            memory_type=MemoryType.FAILURE_AVOIDANCE,
            source_status="success",
            state=MemoryState.ACTIVE,
            confidence=0.9,
            quality_score=0.8,
            created_at="2026-04-01T00:00:00Z",
            dedup_key="failure_avoidance:reproduce-first",
        )
    ]
    candidate = MemoryRecord(
        memory_id="new",
        task_id="t2",
        query="fix failing test",
        content="Patch immediately before looking at the failing test.",
        memory_type=MemoryType.FAILURE_AVOIDANCE,
        source_status="fail",
        state=MemoryState.PROVISIONAL,
        confidence=0.2,
        quality_score=0.2,
        created_at="2026-04-02T00:00:00Z",
        dedup_key="failure_avoidance:reproduce-first",
    )
    new_active, archived, action = resolve_candidate(candidate, active)
    assert action == "DISCARD"
    assert len(new_active) == 1
    assert archived[0].memory_id == "new"


def test_resolve_candidate_merges_same_dedup_key():
    active = [
        MemoryRecord(
            memory_id="old",
            task_id="t1",
            query="fix failing test",
            content="Inspect failing tests before patching.",
            memory_type=MemoryType.PROCEDURAL_HINT,
            source_status="success",
            state=MemoryState.ACTIVE,
            confidence=0.8,
            quality_score=0.8,
            created_at="2026-04-01T00:00:00Z",
            dedup_key="procedural_hint:inspect-tests",
        )
    ]
    candidate = MemoryRecord(
        memory_id="new",
        task_id="t2",
        query="fix failing test",
        content="Inspect the failing regression tests before patching production code.",
        memory_type=MemoryType.PROCEDURAL_HINT,
        source_status="success",
        state=MemoryState.PROVISIONAL,
        confidence=0.9,
        quality_score=0.85,
        created_at="2026-04-02T00:00:00Z",
        dedup_key="procedural_hint:inspect-tests",
    )
    new_active, archived, action = resolve_candidate(candidate, active)
    assert action == "MERGE"
    assert len(new_active) == 1
    assert archived == []
