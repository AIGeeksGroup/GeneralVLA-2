from minisweagent.memory_v2.governance import run_budgeted_governance
from minisweagent.memory_v2.schema import MemoryRecord, MemoryState, MemoryType


def _record(
    memory_id: str,
    *,
    query: str,
    content: str,
    source_status: str = "success",
    use_count: int = 0,
    confidence: float = 0.8,
    quality_score: float = 0.8,
    dedup_key: str | None = None,
    created_at: str = "2026-04-01T00:00:00Z",
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        task_id=f"task-{memory_id}",
        query=query,
        content=content,
        memory_type=MemoryType.PROCEDURAL_HINT,
        source_status=source_status,
        state=MemoryState.ACTIVE,
        confidence=confidence,
        quality_score=quality_score,
        created_at=created_at,
        use_count=use_count,
        dedup_key=dedup_key or f"hint:{memory_id}",
    )


def test_budgeted_governance_summarizes_similar_successes():
    active = [
        _record(
            "a1",
            query="fix astropy failing regression",
            content="Inspect the failing regression test before editing production code.",
            dedup_key="hint:inspect-regression",
        ),
        _record(
            "a2",
            query="fix astropy regression failure",
            content="Read the failing regression test first, then patch the production code.",
            dedup_key="hint:read-regression-first",
        ),
        _record(
            "a3",
            query="debug package failure",
            content="Run the targeted pytest before changing implementation details.",
            dedup_key="hint:run-targeted-pytest",
        ),
    ]
    result = run_budgeted_governance(
        active_records=active,
        summary_records=[],
        now="2026-04-03T00:00:00Z",
        max_active_records=2,
        similarity_threshold=0.35,
        cluster_min_size=2,
        retire_failure_days=14,
    )
    assert len(result["active"]) <= 2
    assert len(result["summary"]) == 1
    assert result["summary"][0].state == MemoryState.SUMMARY
    assert len(result["archived"]) >= 2


def test_budgeted_governance_retires_stale_failed_memory():
    active = [
        _record(
            "fail-old",
            query="fix astropy bug",
            content="Blindly edit setup.py first.",
            source_status="fail",
            confidence=0.2,
            quality_score=0.2,
            created_at="2026-03-01T00:00:00Z",
        ),
        _record(
            "good",
            query="fix astropy bug",
            content="Reproduce the failing test before patching.",
            source_status="success",
            use_count=1,
            confidence=0.9,
            quality_score=0.9,
        ),
    ]
    result = run_budgeted_governance(
        active_records=active,
        summary_records=[],
        now="2026-04-03T00:00:00Z",
        max_active_records=5,
        similarity_threshold=0.35,
        cluster_min_size=2,
        retire_failure_days=14,
    )
    assert [record.memory_id for record in result["active"]] == ["good"]
    assert any(record.memory_id == "fail-old" for record in result["archived"])


def test_budgeted_governance_does_not_summarize_single_task_memories():
    active = [
        _record(
            "a1",
            query="fix astropy regression",
            content="Inspect the failing regression test before editing production code.",
            dedup_key="hint:inspect-regression",
        ),
        _record(
            "a2",
            query="fix astropy regression",
            content="Read the failing regression test first, then patch the production code.",
            dedup_key="hint:read-regression-first",
        ),
        _record(
            "a3",
            query="fix astropy regression",
            content="Run the targeted pytest before changing implementation details.",
            dedup_key="hint:run-targeted-pytest",
        ),
    ]
    active = [record.model_copy(update={"task_id": "same-task"}) for record in active]
    result = run_budgeted_governance(
        active_records=active,
        summary_records=[],
        now="2026-04-03T00:00:00Z",
        max_active_records=6,
        similarity_threshold=0.35,
        cluster_min_size=2,
        retire_failure_days=14,
    )
    assert len(result["summary"]) == 0
    assert len(result["active"]) == 3


def test_budgeted_governance_summarizes_semantically_similar_tooling_memories():
    active = [
        _record(
            "tool-1",
            query="fix astropy transform bug",
            content=(
                "Choose the Right Tool for Code Modification\n"
                "For complex code modifications in Python, programmatic tools are more reliable than shell utilities.\n"
                "Shell commands like `sed` are brittle for indentation-sensitive multi-line edits. Prefer a Python script."
            ),
            dedup_key="tool_usage:choose-right-tool",
        ).model_copy(update={"memory_type": MemoryType.TOOL_USAGE, "task_id": "task-a"}),
        _record(
            "tool-2",
            query="fix astropy table bug",
            content=(
                "Robust Multi-Line Code Modification\n"
                "For complex multi-line Python edits, use a dedicated Python script to replace whole blocks.\n"
                "Text tools like `sed` are error-prone for indentation-sensitive changes."
            ),
            dedup_key="tool_usage:robust-multiline-edit",
        ).model_copy(update={"memory_type": MemoryType.TOOL_USAGE, "task_id": "task-b"}),
    ]
    result = run_budgeted_governance(
        active_records=active,
        summary_records=[],
        now="2026-04-03T00:00:00Z",
        max_active_records=6,
        similarity_threshold=0.35,
        cluster_min_size=2,
        retire_failure_days=14,
    )
    assert len(result["summary"]) == 1
    assert len(result["active"]) == 0
    assert len(result["archived"]) == 2


def test_budgeted_governance_archives_weaker_conflicting_guidance():
    active = [
        _record(
            "good",
            query="fix failing regression",
            content="Do not patch before reproducing the failing test. Reproduce first, then inspect the failure.",
            dedup_key="failure_avoidance:reproduce-before-patch",
            confidence=0.9,
            quality_score=0.9,
        ).model_copy(update={"memory_type": MemoryType.FAILURE_AVOIDANCE, "task_id": "task-a"}),
        _record(
            "bad",
            query="fix failing regression",
            content="Patch immediately before reproducing the failing test.",
            source_status="fail",
            dedup_key="failure_avoidance:patch-before-reproduce",
            confidence=0.2,
            quality_score=0.2,
        ).model_copy(update={"memory_type": MemoryType.FAILURE_AVOIDANCE, "task_id": "task-b"}),
    ]
    result = run_budgeted_governance(
        active_records=active,
        summary_records=[],
        now="2026-04-03T00:00:00Z",
        max_active_records=6,
        similarity_threshold=0.35,
        cluster_min_size=2,
        retire_failure_days=14,
    )
    assert [record.memory_id for record in result["active"]] == ["good"]
    assert result["summary"] == []
    assert any(record.memory_id == "bad" for record in result["archived"])
