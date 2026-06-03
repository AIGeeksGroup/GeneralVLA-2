from pathlib import Path

from minisweagent.memory_v2.schema import MemoryRecord, MemoryState, MemoryType
from minisweagent.memory_v2.store import JsonlMemoryStore


def test_store_separates_states(tmp_path: Path):
    store = JsonlMemoryStore(tmp_path)
    rec = MemoryRecord(
        memory_id="m1",
        task_id="task1",
        query="q",
        content="c",
        memory_type=MemoryType.PROCEDURAL_HINT,
        source_status="success",
        state=MemoryState.PROVISIONAL,
        confidence=0.8,
        quality_score=0.8,
        created_at="2026-04-01T00:00:00Z",
        dedup_key="k1",
    )
    store.add(rec)
    assert len(store.load_active()) == 0
    assert len(store.load_provisional()) == 1
    assert len(store.load_archive()) == 0
    assert len(store.load_summary()) == 0


def test_store_persists_summary_records(tmp_path: Path):
    store = JsonlMemoryStore(tmp_path)
    rec = MemoryRecord(
        memory_id="s1",
        task_id="task1",
        query="q",
        content="cluster summary",
        memory_type=MemoryType.PROCEDURAL_HINT,
        source_status="success",
        state=MemoryState.SUMMARY,
        confidence=0.9,
        quality_score=0.85,
        created_at="2026-04-01T00:00:00Z",
        dedup_key="summary:k1",
    )
    store.add(rec)
    summaries = store.load_summary()
    assert len(summaries) == 1
    assert summaries[0].memory_id == "s1"
