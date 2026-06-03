#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_reports import compare_summaries, summarize_memory_root, summarize_results_dir, summarize_results_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and consolidated-memory SWE runs.")
    parser.add_argument(
        "--candidate-results-dir",
        type=Path,
        action="append",
        required=True,
        help="Results dir for the candidate run. Repeat to merge retries.",
    )
    parser.add_argument(
        "--baseline-results-dir",
        type=Path,
        action="append",
        help="Optional baseline results dir. Repeat to merge retries.",
    )
    parser.add_argument(
        "--candidate-official-report",
        type=Path,
        help="Optional sb-cli JSON report for the candidate run.",
    )
    parser.add_argument(
        "--baseline-official-report",
        type=Path,
        help="Optional sb-cli JSON report for the baseline run.",
    )
    parser.add_argument("--memory-root", type=Path, help="Optional memory root for the candidate run")
    args = parser.parse_args()

    if args.baseline_results_dir:
        baseline = (
            summarize_results_dir(args.baseline_results_dir[0], official_report_path=args.baseline_official_report)
            if len(args.baseline_results_dir) == 1
            else summarize_results_dirs(args.baseline_results_dir, official_report_path=args.baseline_official_report)
        )
    else:
        baseline = None
    candidate = (
        summarize_results_dir(args.candidate_results_dir[0], official_report_path=args.candidate_official_report)
        if len(args.candidate_results_dir) == 1
        else summarize_results_dirs(args.candidate_results_dir, official_report_path=args.candidate_official_report)
    )
    memory = summarize_memory_root(args.memory_root) if args.memory_root else None
    print(json.dumps(compare_summaries(baseline=baseline, candidate=candidate, memory=memory), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
