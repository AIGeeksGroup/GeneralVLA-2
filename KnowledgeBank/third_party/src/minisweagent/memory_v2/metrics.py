from __future__ import annotations

from collections import Counter

from .schema import MemoryRecord


def compute_bank_growth_rate(series: list[int]) -> float:
    if len(series) < 2:
        return 0.0
    start = series[0]
    end = series[-1]
    if start == 0:
        return float(end)
    return (end - start) / start


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def count_duplicates(records: list[MemoryRecord]) -> int:
    counter = Counter(record.dedup_key for record in records)
    return sum(count - 1 for count in counter.values() if count > 1)


def count_conflicted_records(records: list[MemoryRecord]) -> int:
    return sum(1 for record in records if record.conflicts_with)


def collect_memory_health(
    *,
    active_records: list[MemoryRecord],
    summary_records: list[MemoryRecord] | None = None,
    provisional_records: list[MemoryRecord],
    archive_records: list[MemoryRecord],
    active_history: list[int] | None = None,
) -> dict[str, float | int]:
    active_history = active_history or [len(active_records)]
    summary_records = summary_records or []
    return {
        "active_records": len(active_records),
        "summary_records": len(summary_records),
        "provisional_records": len(provisional_records),
        "archive_records": len(archive_records),
        "duplicate_active_records": count_duplicates(active_records),
        "conflicted_active_records": count_conflicted_records(active_records),
        "failed_active_records": sum(1 for record in active_records if record.source_status != "success"),
        "active_growth_rate": compute_bank_growth_rate(active_history),
        "duplicate_active_ratio": safe_ratio(count_duplicates(active_records), len(active_records)),
        "conflicted_active_ratio": safe_ratio(count_conflicted_records(active_records), len(active_records)),
    }
