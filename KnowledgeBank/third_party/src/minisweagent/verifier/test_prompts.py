from minisweagent.verifier.prompts import build_single_trajectory_prompt
from minisweagent.verifier.swe_criteria import SWE_CRITERIA


def test_single_trajectory_prompt_contains_task_trace_patch_and_one_criterion():
    prompt = build_single_trajectory_prompt(
        task="Fix a parser regression.",
        trajectory="THOUGHT: I found the parser.\n[Output]\npytest passed",
        patch="diff --git a/parser.py b/parser.py",
        exit_status="Submitted",
        criterion=SWE_CRITERIA[0],
    )

    assert "Fix a parser regression." in prompt
    assert "THOUGHT: I found the parser." in prompt
    assert "diff --git a/parser.py b/parser.py" in prompt
    assert "Submitted" in prompt
    assert SWE_CRITERIA[0].name in prompt
    assert "<score>" in prompt
    assert "<score_A>" not in prompt
    assert "<score_B>" not in prompt
