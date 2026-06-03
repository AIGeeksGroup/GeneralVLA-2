from minisweagent.memory_v2.retrieval import rank_memory_record, score_memory_record
from minisweagent.memory_v2.schema import MemoryRecord, MemoryState, MemoryType


def test_score_penalizes_conflicted_stale_memory():
    stale = score_memory_record(
        relevance=0.9,
        confidence=0.3,
        success_prior=0.0,
        recency_bonus=0.0,
        usage_bonus=0.0,
        conflict_penalty=0.6,
        staleness_penalty=0.4,
    )
    fresh = score_memory_record(
        relevance=0.8,
        confidence=0.9,
        success_prior=0.2,
        recency_bonus=0.2,
        usage_bonus=0.1,
        conflict_penalty=0.0,
        staleness_penalty=0.0,
    )
    assert fresh > stale


def test_summary_memory_gets_ranked_but_below_equivalent_active_memory():
    active = MemoryRecord(
        memory_id="a1",
        task_id="t1",
        query="fix astropy regression",
        content="Inspect the failing regression test before patching.",
        memory_type=MemoryType.PROCEDURAL_HINT,
        source_status="success",
        state=MemoryState.ACTIVE,
        confidence=0.9,
        quality_score=0.9,
        created_at="2026-04-01T00:00:00Z",
        dedup_key="hint:inspect-regression",
    )
    summary = active.model_copy(
        update={
            "memory_id": "s1",
            "state": MemoryState.SUMMARY,
            "dedup_key": "summary:inspect-regression",
            "content": "Summary:\n- Inspect the failing regression test before patching.",
        }
    )
    active_rank = rank_memory_record(active, "fix astropy regression", now="2026-04-03T00:00:00Z")
    summary_rank = rank_memory_record(summary, "fix astropy regression", now="2026-04-03T00:00:00Z")
    assert active_rank.score > summary_rank.score


def test_rank_memory_penalizes_uncertain_low_verifier_score():
    strong = MemoryRecord(
        memory_id="strong",
        task_id="t1",
        query="fix astropy regression in table validation",
        content="Inspect astropy table validation and patch the failing regression.",
        memory_type=MemoryType.PROCEDURAL_HINT,
        source_status="verified_success",
        state=MemoryState.ACTIVE,
        confidence=0.8,
        quality_score=0.8,
        created_at="2026-04-01T00:00:00Z",
        dedup_key="procedural_hint:strong",
        verifier_score=0.8,
        verifier_label="verified_success",
    )
    uncertain = strong.model_copy(
        update={
            "memory_id": "uncertain",
            "source_status": "uncertain",
            "confidence": 0.95,
            "quality_score": 0.95,
            "dedup_key": "procedural_hint:uncertain",
            "verifier_score": 0.49,
            "verifier_label": "uncertain",
        }
    )

    strong_rank = rank_memory_record(strong, "fix astropy regression in table validation", now="2026-04-03T00:00:00Z")
    uncertain_rank = rank_memory_record(uncertain, "fix astropy regression in table validation", now="2026-04-03T00:00:00Z")

    assert strong_rank.score > uncertain_rank.score
