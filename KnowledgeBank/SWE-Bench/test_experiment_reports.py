from __future__ import annotations

import json
from pathlib import Path
import pytest

from experiment_reports import (
    compare_summaries,
    summarize_memory_root,
    summarize_official_report,
    summarize_results_dir,
    summarize_results_dirs,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summarize_results_dir(tmp_path: Path):
    results_dir = tmp_path / "results"
    _write_json(
        results_dir / "preds.json",
        {
            "task-1": {"model_name_or_path": "demo", "instance_id": "task-1", "model_patch": ""},
            "task-2": {"model_name_or_path": "demo", "instance_id": "task-2", "model_patch": ""},
        },
    )
    _write_json(
        results_dir / "task-1" / "task-1.traj.json",
        {"info": {"exit_status": "Submitted", "model_stats": {"api_calls": 2, "instance_cost": 0.5}}},
    )
    _write_json(
        results_dir / "task-2" / "task-2.traj.json",
        {"info": {"exit_status": "FormatError", "model_stats": {"api_calls": 4, "instance_cost": 1.5}}},
    )

    summary = summarize_results_dir(results_dir)
    assert summary["num_predictions"] == 2
    assert summary["num_trajectories"] == 2
    assert summary["submitted_count"] == 1
    assert summary["avg_api_calls"] == 3.0


def test_summarize_results_dir_reports_average_steps_from_assistant_messages(tmp_path: Path):
    results_dir = tmp_path / "results"
    _write_json(
        results_dir / "preds.json",
        {"task-1": {"model_name_or_path": "demo", "instance_id": "task-1", "model_patch": ""}},
    )
    _write_json(
        results_dir / "task-1" / "task-1.traj.json",
        {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "u3"},
                {"role": "assistant", "content": "a3"},
            ],
            "info": {
                "exit_status": "Submitted",
                "model_stats": {"api_calls": 99, "instance_cost": 0.5},
            },
        },
    )

    summary = summarize_results_dir(results_dir)

    assert summary["AS"] == 3.0
    assert summary["avg_steps"] == 3.0


def test_summarize_official_report_exposes_resolve_rate_fields(tmp_path: Path):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "total_instances": 500,
                "submitted_instances": 30,
                "completed_instances": 28,
                "pending_instances": 2,
                "resolved_instances": 9,
                "failed_instances": 1,
                "error_instances": 0,
            }
        ),
        encoding="utf-8",
    )

    report = summarize_official_report(report_path)

    assert report["resolve_rate"] == pytest.approx(9 / 500)
    assert report["resolve_rate_submitted"] == pytest.approx(9 / 30)
    assert report["pending_instances"] == 2
    assert report["report_complete"] is False


def test_summarize_memory_root_and_compare(tmp_path: Path):
    memory_root = tmp_path / "memory"
    metrics_path = memory_root / "metrics" / "memory_health.jsonl"
    edit_events_path = memory_root / "logs" / "edit_events.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps({"active_records": 1, "duplicate_active_ratio": 0.0, "conflicted_active_ratio": 0.0, "active_growth_rate": 0.0, "archive_records": 0, "failed_active_records": 0}),
                json.dumps({"active_records": 2, "duplicate_active_ratio": 0.25, "conflicted_active_ratio": 0.1, "active_growth_rate": 1.0, "archive_records": 1, "failed_active_records": 0}),
            ]
        ),
        encoding="utf-8",
    )
    edit_events_path.parent.mkdir(parents=True, exist_ok=True)
    edit_events_path.write_text(
        "\n".join(
            [
                json.dumps({"source_status": "verified_success", "verifier_score": 0.8, "candidate_count": 1}),
                json.dumps({"source_status": "verified_fail", "verifier_score": 0.2, "candidate_count": 1}),
                json.dumps({"event": "memory_write_abstained", "verifier_label": "uncertain", "verifier_score": 0.5}),
            ]
        ),
        encoding="utf-8",
    )
    memory = summarize_memory_root(memory_root)
    comparison = compare_summaries(
        baseline={"submitted_ratio": 0.4, "avg_api_calls": 12.0, "avg_instance_cost": 1.2},
        candidate={"submitted_ratio": 0.5, "avg_api_calls": 10.0, "avg_instance_cost": 1.0},
        memory=memory,
    )
    assert memory["last_duplicate_ratio"] == 0.25
    assert memory["verifier_score_mean"] == pytest.approx(0.5)
    assert memory["verified_success_rate"] == pytest.approx(1 / 3)
    assert memory["uncertain_write_abstain_rate"] == pytest.approx(1 / 3)
    assert comparison["deltas"]["submitted_ratio_delta"] == pytest.approx(0.1)
    assert comparison["deltas"]["avg_api_calls_delta"] == -2.0


def test_summarize_results_dirs_merges_retry_runs_by_instance_id(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    _write_json(
        first / "preds.json",
        {"task-1": {"model_name_or_path": "demo", "instance_id": "task-1", "model_patch": "old"}},
    )
    _write_json(
        first / "task-1" / "task-1.traj.json",
        {"instance_id": "task-1", "info": {"exit_status": "CalledProcessError", "model_stats": {"api_calls": 2, "instance_cost": 0.5}}},
    )

    _write_json(
        second / "preds.json",
        {"task-1": {"model_name_or_path": "demo", "instance_id": "task-1", "model_patch": "new"}},
    )
    _write_json(
        second / "task-1" / "task-1.traj.json",
        {"instance_id": "task-1", "info": {"exit_status": "Submitted", "model_stats": {"api_calls": 5, "instance_cost": 1.25}}},
    )

    summary = summarize_results_dirs([first, second])

    assert summary["num_predictions"] == 1
    assert summary["num_trajectories"] == 1
    assert summary["submitted_count"] == 1
    assert summary["avg_api_calls"] == 5.0
