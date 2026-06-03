from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class VerificationLabel(str, Enum):
    VERIFIED_SUCCESS = "verified_success"
    VERIFIED_FAIL = "verified_fail"
    UNCERTAIN = "uncertain"


@dataclass(slots=True)
class VerificationResult:
    label: VerificationLabel
    score: float
    confidence: float
    criteria_scores: dict[str, float]
    raw_outputs: list[str] = field(default_factory=list)
    model_name: str = ""
    n_reps: int = 1
    used_logprobs: bool = False
    exit_status: str | None = None

    @classmethod
    def from_scores(
        cls,
        *,
        criteria_scores: dict[str, float],
        raw_outputs: list[str],
        model_name: str,
        n_reps: int,
        used_logprobs: bool,
        exit_status: str | None = None,
        success_threshold: float = 0.70,
        fail_threshold: float = 0.35,
    ) -> "VerificationResult":
        score = _mean(criteria_scores.values())
        if exit_status is not None and exit_status != "Submitted":
            score = min(score, 0.49)
        if score >= success_threshold:
            label = VerificationLabel.VERIFIED_SUCCESS
        elif score <= fail_threshold:
            label = VerificationLabel.VERIFIED_FAIL
        else:
            label = VerificationLabel.UNCERTAIN
        confidence = abs(score - 0.5) * 2.0
        return cls(
            label=label,
            score=round(score, 4),
            confidence=round(confidence, 4),
            criteria_scores={key: round(value, 4) for key, value in criteria_scores.items()},
            raw_outputs=raw_outputs,
            model_name=model_name,
            n_reps=n_reps,
            used_logprobs=used_logprobs,
            exit_status=exit_status,
        )


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.5
    return sum(values) / len(values)
