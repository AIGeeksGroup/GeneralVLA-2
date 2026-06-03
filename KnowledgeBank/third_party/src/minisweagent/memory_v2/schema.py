from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MemoryState(str, Enum):
    PROVISIONAL = "provisional"
    ACTIVE = "active"
    SUMMARY = "summary"
    ARCHIVE = "archive"


class MemoryType(str, Enum):
    PROCEDURAL_HINT = "procedural_hint"
    FAILURE_AVOIDANCE = "failure_avoidance"
    TOOL_USAGE = "tool_usage"


class MemoryRecord(BaseModel):
    memory_id: str
    task_id: str
    query: str
    content: str
    memory_type: MemoryType
    source_status: str
    state: MemoryState
    confidence: float
    quality_score: float
    created_at: str
    last_used_at: Optional[str] = None
    use_count: int = 0
    dedup_key: str
    supersedes: list[str] = Field(default_factory=list)
    conflicts_with: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None
    verifier_score: Optional[float] = None
    verifier_confidence: Optional[float] = None
    verifier_label: Optional[str] = None
    verifier_criteria: dict[str, float] = Field(default_factory=dict)
    verifier_model: Optional[str] = None
