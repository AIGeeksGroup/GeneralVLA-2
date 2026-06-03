from __future__ import annotations

from .prompts import build_single_trajectory_prompt
from .scales import score_from_text
from .schema import VerificationResult
from .swe_criteria import SWE_CRITERIA, VerificationCriterion


def verify_trajectory(
    *,
    task: str,
    trajectory: str,
    patch: str,
    exit_status: str | None,
    client,
    criteria: list[VerificationCriterion] | None = None,
    n_reps: int = 4,
    model_name: str = "",
    success_threshold: float = 0.70,
    fail_threshold: float = 0.35,
) -> VerificationResult:
    criteria = criteria or SWE_CRITERIA
    criterion_scores: dict[str, list[float]] = {criterion.id: [] for criterion in criteria}
    raw_outputs: list[str] = []
    used_logprobs = False

    for criterion in criteria:
        for _ in range(n_reps):
            prompt = build_single_trajectory_prompt(
                task=task,
                trajectory=trajectory,
                patch=patch,
                exit_status=exit_status or "unknown",
                criterion=criterion,
            )
            response = client.score_prompt(prompt)
            raw_text = response.get("text", "")
            raw_outputs.append(raw_text)
            score = response.get("score")
            if score is None:
                score = score_from_text(raw_text)
            else:
                used_logprobs = bool(response.get("used_logprobs", False)) or used_logprobs
            criterion_scores[criterion.id].append(float(score))

    averaged = {
        criterion_id: sum(scores) / len(scores) if scores else 0.5
        for criterion_id, scores in criterion_scores.items()
    }
    return VerificationResult.from_scores(
        criteria_scores=averaged,
        raw_outputs=raw_outputs,
        model_name=model_name,
        n_reps=n_reps,
        used_logprobs=used_logprobs,
        exit_status=exit_status,
        success_threshold=success_threshold,
        fail_threshold=fail_threshold,
    )
