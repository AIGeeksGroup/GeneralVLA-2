#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_snapshots(snapshots: list[dict]) -> dict:
    if not snapshots:
        return {
            "num_snapshots": 0,
            "first_active_records": 0,
            "last_active_records": 0,
            "peak_active_records": 0,
            "first_summary_records": 0,
            "last_summary_records": 0,
            "peak_summary_records": 0,
            "first_duplicate_ratio": 0.0,
            "last_duplicate_ratio": 0.0,
            "first_conflicted_ratio": 0.0,
            "last_conflicted_ratio": 0.0,
            "last_growth_rate": 0.0,
            "last_archive_records": 0,
            "last_failed_active_records": 0,
        }
    first = snapshots[0]
    last = snapshots[-1]
    return {
        "num_snapshots": len(snapshots),
        "first_active_records": first.get("active_records", 0),
        "last_active_records": last.get("active_records", 0),
        "peak_active_records": max(snapshot.get("active_records", 0) for snapshot in snapshots),
        "first_summary_records": first.get("summary_records", 0),
        "last_summary_records": last.get("summary_records", 0),
        "peak_summary_records": max(snapshot.get("summary_records", 0) for snapshot in snapshots),
        "first_duplicate_ratio": first.get("duplicate_active_ratio", 0.0),
        "last_duplicate_ratio": last.get("duplicate_active_ratio", 0.0),
        "first_conflicted_ratio": first.get("conflicted_active_ratio", 0.0),
        "last_conflicted_ratio": last.get("conflicted_active_ratio", 0.0),
        "last_growth_rate": last.get("active_growth_rate", 0.0),
        "last_archive_records": last.get("archive_records", 0),
        "last_failed_active_records": last.get("failed_active_records", 0),
    }


def summarize_verifier_events(events: list[dict]) -> dict:
    verifier_events = [
        event
        for event in events
        if "verifier_score" in event or "verifier_label" in event or str(event.get("source_status", "")).startswith("verified_")
    ]
    if not verifier_events:
        return {
            "num_verifier_events": 0,
            "verifier_score_mean": 0.0,
            "verified_success_rate": 0.0,
            "verified_fail_rate": 0.0,
            "uncertain_rate": 0.0,
            "uncertain_write_abstain_rate": 0.0,
            "memory_generated_from_low_score_rate": 0.0,
        }
    scores = [event["verifier_score"] for event in verifier_events if isinstance(event.get("verifier_score"), int | float)]
    labels = [
        event.get("verifier_label") or event.get("source_status")
        for event in verifier_events
    ]
    label_counter = Counter(label for label in labels if label)
    total = len(verifier_events)
    low_score_generated = sum(
        1
        for event in verifier_events
        if event.get("candidate_count", 0) > 0 and event.get("verifier_score") is not None and event["verifier_score"] < 0.55
    )
    return {
        "num_verifier_events": total,
        "verifier_score_mean": (sum(scores) / len(scores)) if scores else 0.0,
        "verified_success_rate": label_counter.get("verified_success", 0) / total,
        "verified_fail_rate": label_counter.get("verified_fail", 0) / total,
        "uncertain_rate": label_counter.get("uncertain", 0) / total,
        "uncertain_write_abstain_rate": sum(1 for event in verifier_events if event.get("event") == "memory_write_abstained") / total,
        "memory_generated_from_low_score_rate": low_score_generated / total,
    }


def summarize_memory_root(memory_root: Path) -> dict:
    metrics_path = memory_root / "metrics" / "memory_health.jsonl"
    edit_events_path = memory_root / "logs" / "edit_events.jsonl"
    snapshots = read_jsonl(metrics_path)
    summary = summarize_snapshots(snapshots)
    summary.update(summarize_verifier_events(read_jsonl(edit_events_path)))
    summary["memory_root"] = str(memory_root)
    summary["metrics_path"] = str(metrics_path)
    summary["edit_events_path"] = str(edit_events_path)
    return summary


def _iter_trajectory_paths(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*/*.traj.json"))


def _instance_id_from_trajectory(path: Path, trajectory: dict) -> str:
    instance_id = trajectory.get("instance_id")
    if instance_id:
        return instance_id
    return path.parent.name


def _assistant_steps(trajectory: dict) -> int:
    messages = trajectory.get("messages")
    if isinstance(messages, list) and messages:
        return sum(1 for message in messages if message.get("role") == "assistant")
    return trajectory.get("info", {}).get("model_stats", {}).get("api_calls", 0)


def summarize_official_report(report_path: Path) -> dict:
    report = read_json(report_path)
    total_instances = report.get("total_instances", 0)
    submitted_instances = report.get("submitted_instances", 0)
    completed_instances = report.get("completed_instances", 0)
    pending_instances = report.get("pending_instances", 0)
    resolved_instances = report.get("resolved_instances", 0)
    failed_instances = report.get("failed_instances", 0)
    error_instances = report.get("error_instances", 0)
    return {
        "official_report_path": str(report_path),
        "total_instances": total_instances,
        "submitted_instances": submitted_instances,
        "completed_instances": completed_instances,
        "pending_instances": pending_instances,
        "resolved_instances": resolved_instances,
        "failed_instances": failed_instances,
        "error_instances": error_instances,
        "resolve_rate": (resolved_instances / total_instances) if total_instances else 0.0,
        "resolve_rate_submitted": (resolved_instances / submitted_instances) if submitted_instances else 0.0,
        "report_complete": pending_instances == 0,
    }


def _summary_from_records(
    *,
    results_dir: str,
    preds: dict,
    trajectories: list[dict],
    official_report: dict | None = None,
) -> dict:
    exit_status_counter = Counter(
        trajectory.get("info", {}).get("exit_status", "unknown") for trajectory in trajectories
    )
    api_calls = [trajectory.get("info", {}).get("model_stats", {}).get("api_calls", 0) for trajectory in trajectories]
    steps = [_assistant_steps(trajectory) for trajectory in trajectories]
    costs = [trajectory.get("info", {}).get("model_stats", {}).get("instance_cost", 0.0) for trajectory in trajectories]
    summary = {
        "results_dir": str(results_dir),
        "num_predictions": len(preds),
        "num_trajectories": len(trajectories),
        "submitted_count": exit_status_counter.get("Submitted", 0),
        "submitted_ratio": (exit_status_counter.get("Submitted", 0) / len(trajectories)) if trajectories else 0.0,
        "avg_api_calls": (sum(api_calls) / len(api_calls)) if api_calls else 0.0,
        "avg_steps": (sum(steps) / len(steps)) if steps else 0.0,
        "AS": (sum(steps) / len(steps)) if steps else 0.0,
        "avg_instance_cost": (sum(costs) / len(costs)) if costs else 0.0,
        "total_api_calls": sum(api_calls),
        "total_steps": sum(steps),
        "total_instance_cost": sum(costs),
        "exit_status_counts": dict(exit_status_counter),
    }
    if official_report is not None:
        summary.update(official_report)
    return summary


def summarize_results_dir(results_dir: Path, official_report_path: Path | None = None) -> dict:
    preds_path = results_dir / "preds.json"
    preds = read_json(preds_path) if preds_path.exists() else {}
    trajectories = [read_json(path) for path in _iter_trajectory_paths(results_dir)]
    official_report = summarize_official_report(official_report_path) if official_report_path else None
    return _summary_from_records(
        results_dir=str(results_dir),
        preds=preds,
        trajectories=trajectories,
        official_report=official_report,
    )


def summarize_results_dirs(results_dirs: list[Path], official_report_path: Path | None = None) -> dict:
    merged_preds: dict[str, dict] = {}
    merged_trajectories: dict[str, dict] = {}

    for results_dir in results_dirs:
        preds_path = results_dir / "preds.json"
        preds = read_json(preds_path) if preds_path.exists() else {}
        merged_preds.update(preds)

        for path in _iter_trajectory_paths(results_dir):
            trajectory = read_json(path)
            merged_trajectories[_instance_id_from_trajectory(path, trajectory)] = trajectory

    label = ",".join(str(path) for path in results_dirs)
    official_report = summarize_official_report(official_report_path) if official_report_path else None
    return _summary_from_records(
        results_dir=label,
        preds=merged_preds,
        trajectories=list(merged_trajectories.values()),
        official_report=official_report,
    )


def compare_summaries(*, baseline: dict | None, candidate: dict, memory: dict | None = None) -> dict:
    comparison = {
        "candidate": candidate,
        "memory": memory,
    }
    if baseline is None:
        return comparison
    comparison["baseline"] = baseline
    comparison["deltas"] = {
        "submitted_ratio_delta": candidate.get("submitted_ratio", 0.0) - baseline.get("submitted_ratio", 0.0),
        "avg_api_calls_delta": candidate.get("avg_api_calls", 0.0) - baseline.get("avg_api_calls", 0.0),
        "avg_steps_delta": candidate.get("avg_steps", 0.0) - baseline.get("avg_steps", 0.0),
        "avg_instance_cost_delta": candidate.get("avg_instance_cost", 0.0) - baseline.get("avg_instance_cost", 0.0),
    }
    if "resolve_rate" in baseline and "resolve_rate" in candidate:
        comparison["deltas"]["resolve_rate_delta"] = candidate.get("resolve_rate", 0.0) - baseline.get("resolve_rate", 0.0)
    if "resolve_rate_submitted" in baseline and "resolve_rate_submitted" in candidate:
        comparison["deltas"]["resolve_rate_submitted_delta"] = (
            candidate.get("resolve_rate_submitted", 0.0) - baseline.get("resolve_rate_submitted", 0.0)
        )
    if memory is not None:
        comparison["deltas"] |= {
            "memory_growth_rate": memory.get("last_growth_rate", 0.0),
            "memory_duplicate_ratio": memory.get("last_duplicate_ratio", 0.0),
            "memory_conflict_ratio": memory.get("last_conflicted_ratio", 0.0),
        }
    return comparison
