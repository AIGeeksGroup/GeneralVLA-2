from __future__ import annotations

from .scales import SCALE_DESCRIPTION, SCORE_TAG
from .swe_criteria import GROUND_TRUTH_NOTE, VerificationCriterion


def build_single_trajectory_prompt(
    *,
    task: str,
    trajectory: str,
    patch: str,
    exit_status: str,
    criterion: VerificationCriterion,
) -> str:
    patch_text = patch.strip() if patch and patch.strip() else "(no final patch/result provided)"
    return (
        "You are an expert evaluator of AI coding agents on SWE-Bench tasks. "
        f"Evaluate the trajectory on ONE specific criterion: **{criterion.name}**.\n\n"
        f"{GROUND_TRUTH_NOTE}\n\n"
        f"**Task:**\n{task}\n\n"
        f"**Trajectory:**\n{trajectory}\n\n"
        f"**Final Patch or Result:**\n{patch_text}\n\n"
        f"**Exit Status:** {exit_status}\n\n"
        f"**Evaluation Guideline - {criterion.name}:**\n{criterion.description}\n\n"
        f"Score this trajectory ONLY on {criterion.name}. Ignore aspects not relevant to this criterion.\n\n"
        f"**Rating Scale:**\n{SCALE_DESCRIPTION}\n\n"
        "Think briefly, then output exactly one final score token in this format:\n"
        f"<{SCORE_TAG}>LETTER_A_TO_T</{SCORE_TAG}>"
    )
