# Copyright 2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_reports import summarize_results_dir, summarize_results_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a SWE-Bench run directory.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        required=True,
        help="Directory containing preds.json and trajectories. Repeat to merge retry runs.",
    )
    parser.add_argument(
        "--official-report",
        type=Path,
        help="Optional sb-cli JSON report to merge official Resolve Rate fields into the summary.",
    )
    args = parser.parse_args()
    if len(args.results_dir) == 1:
        summary = summarize_results_dir(args.results_dir[0], official_report_path=args.official_report)
    else:
        summary = summarize_results_dirs(args.results_dir, official_report_path=args.official_report)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
