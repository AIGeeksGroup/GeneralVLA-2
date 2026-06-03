from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VerificationCriterion:
    id: str
    name: str
    description: str


GROUND_TRUTH_NOTE = (
    "Do NOT trust the agent's self-assessment or claims that the patch looks correct. "
    "Agents routinely declare success on patches that fix the wrong file, address only a symptom, "
    "or are subtly broken."
)

SWE_CRITERIA = [
    VerificationCriterion(
        id="root_cause",
        name="Root Cause Analysis",
        description=(
            "Read the issue, identify the buggy behavior it describes, and trace it to the code "
            "that produces it. Decide whether the patch modifies the actual code path responsible "
            "for the bug, or only its symptoms. A patch that edits the buggy function or branch "
            "should score HIGH; a patch that special-cases the issue example or works around the "
            "buggy callee should score LOW."
        ),
    ),
    VerificationCriterion(
        id="code_review",
        name="Code Quality",
        description=(
            "Review the final patch as an experienced code reviewer would. Check syntactic validity, "
            "semantic correctness, preservation of existing contracts, and consistency with the "
            "surrounding code. Penalize silent regressions in code paths the issue did not mention."
        ),
    ),
    VerificationCriterion(
        id="empirical_verification",
        name="Empirical Verification",
        description=(
            "Look at the commands the agent actually ran and what they printed, not what the agent "
            "claimed. Reward trajectories that reproduced the failure, observed the expected behavior "
            "after the fix, and ran relevant existing tests. Penalize agents that declared success "
            "without verification or edited code again after the last successful check."
        ),
    ),
]
