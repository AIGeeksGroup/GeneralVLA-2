#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_reports import summarize_memory_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize KnowledgeBank-CM memory health metrics.")
    parser.add_argument("--memory-root", type=Path, required=True, help="Model-specific memory directory")
    args = parser.parse_args()

    summary = summarize_memory_root(args.memory_root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
