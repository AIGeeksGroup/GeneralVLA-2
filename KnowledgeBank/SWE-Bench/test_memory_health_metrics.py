from minisweagent.memory_v2.metrics import collect_memory_health, compute_bank_growth_rate
from minisweagent.memory_v2.schema import MemoryRecord, MemoryState, MemoryType


def test_bank_growth_rate_is_zero_for_stable_bank():
    assert compute_bank_growth_rate([10, 10, 10]) == 0.0


def test_collect_memory_health_counts_duplicates_and_conflicts():
    active = [
        MemoryRecord(
            memory_id="m1",
            task_id="t1",
            query="q1",
            content="Inspect failing tests before patching.",
            memory_type=MemoryType.PROCEDURAL_HINT,
            source_status="success",
            state=MemoryState.ACTIVE,
            confidence=0.8,
            quality_score=0.8,
            created_at="2026-04-01T00:00:00Z",
            dedup_key="procedural:inspect-tests",
            conflicts_with=[],
        ),
        MemoryRecord(
            memory_id="m2",
            task_id="t2",
            query="q2",
            content="Inspect failing tests before patching production code.",
            memory_type=MemoryType.PROCEDURAL_HINT,
            source_status="fail",
            state=MemoryState.ACTIVE,
            confidence=0.4,
            quality_score=0.4,
            created_at="2026-04-01T00:00:00Z",
            dedup_key="procedural:inspect-tests",
            conflicts_with=["m1"],
        ),
    ]
    snapshot = collect_memory_health(
        active_records=active,
        summary_records=[],
        provisional_records=[],
        archive_records=[],
        active_history=[1, 2],
    )
    assert snapshot["duplicate_active_records"] == 1
    assert snapshot["conflicted_active_records"] == 1


def test_collect_memory_health_reports_summary_count():
    summary = [
        MemoryRecord(
            memory_id="s1",
            task_id="t1",
            query="q1",
            content="Summary:\n- Inspect tests first.",
            memory_type=MemoryType.PROCEDURAL_HINT,
            source_status="success",
            state=MemoryState.SUMMARY,
            confidence=0.9,
            quality_score=0.9,
            created_at="2026-04-01T00:00:00Z",
            dedup_key="summary:inspect-tests",
            conflicts_with=[],
        )
    ]
    snapshot = collect_memory_health(
        active_records=[],
        summary_records=summary,
        provisional_records=[],
        archive_records=[],
        active_history=[0, 0],
    )
    assert snapshot["summary_records"] == 1
