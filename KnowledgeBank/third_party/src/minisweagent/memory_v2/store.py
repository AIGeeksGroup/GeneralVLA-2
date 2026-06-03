from __future__ import annotations

import json
from pathlib import Path

from .schema import MemoryRecord, MemoryState


class JsonlMemoryStore:
    """Simple state-separated memory store for early CM experiments."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = {
            MemoryState.ACTIVE: self.root / "active.jsonl",
            MemoryState.PROVISIONAL: self.root / "provisional.jsonl",
            MemoryState.SUMMARY: self.root / "summary.jsonl",
            MemoryState.ARCHIVE: self.root / "archive.jsonl",
        }
        for path in self.paths.values():
            path.touch(exist_ok=True)

    def add(self, rec: MemoryRecord) -> None:
        with self.paths[rec.state].open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")

    def add_many(self, records: list[MemoryRecord]) -> None:
        for record in records:
            self.add(record)

    def _load(self, state: MemoryState) -> list[MemoryRecord]:
        out: list[MemoryRecord] = []
        with self.paths[state].open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(MemoryRecord(**json.loads(line)))
        return out

    def _rewrite(self, state: MemoryState, records: list[MemoryRecord]) -> None:
        with self.paths[state].open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")

    def load_active(self) -> list[MemoryRecord]:
        return self._load(MemoryState.ACTIVE)

    def load_provisional(self) -> list[MemoryRecord]:
        return self._load(MemoryState.PROVISIONAL)

    def load_archive(self) -> list[MemoryRecord]:
        return self._load(MemoryState.ARCHIVE)

    def load_summary(self) -> list[MemoryRecord]:
        return self._load(MemoryState.SUMMARY)

    def replace_state(self, state: MemoryState, records: list[MemoryRecord]) -> None:
        self._rewrite(state, records)
