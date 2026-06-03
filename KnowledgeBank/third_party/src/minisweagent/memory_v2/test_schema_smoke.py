from minisweagent.memory_v2.schema import MemoryRecord, MemoryState, MemoryType


def test_memory_record_round_trip():
    rec = MemoryRecord(
        memory_id="m1",
        task_id="astropy__astropy-12907",
        query="Fix a failing test around unit conversion.",
        content="Before changing implementation, inspect existing regression tests and reproduction steps.",
        memory_type=MemoryType.PROCEDURAL_HINT,
        source_status="success",
        state=MemoryState.PROVISIONAL,
        confidence=0.9,
        quality_score=0.8,
        created_at="2026-04-01T00:00:00Z",
        last_used_at=None,
        use_count=0,
        dedup_key="procedural:inspect-tests-first",
        supersedes=[],
        conflicts_with=[],
        embedding=None,
    )
    payload = rec.model_dump()
    restored = MemoryRecord(**payload)
    assert restored.memory_id == "m1"
    assert restored.state == MemoryState.PROVISIONAL


def test_memory_record_verifier_fields_default_for_legacy_jsonl():
    restored = MemoryRecord(
        memory_id="m-legacy",
        task_id="task-1",
        query="fix bug",
        content="Inspect the failing path.",
        memory_type=MemoryType.PROCEDURAL_HINT,
        source_status="success",
        state=MemoryState.ACTIVE,
        confidence=0.8,
        quality_score=0.8,
        created_at="2026-04-01T00:00:00Z",
        dedup_key="procedural_hint:inspect-failing-path",
    )

    assert restored.verifier_score is None
    assert restored.verifier_confidence is None
    assert restored.verifier_label is None
    assert restored.verifier_criteria == {}
    assert restored.verifier_model is None
