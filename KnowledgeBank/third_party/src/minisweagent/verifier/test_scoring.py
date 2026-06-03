from minisweagent.verifier.schema import VerificationLabel
from minisweagent.verifier.scoring import verify_trajectory
from minisweagent.verifier.swe_criteria import SWE_CRITERIA


class FakeVerifierClient:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.prompts = []

    def score_prompt(self, prompt):
        self.prompts.append(prompt)
        return {"text": f"<score>{self.tokens.pop(0)}</score>", "score": None, "used_logprobs": False}


def test_verify_trajectory_averages_criteria_and_repetitions():
    client = FakeVerifierClient(["A", "T", "A", "T", "A", "T"])
    result = verify_trajectory(
        task="Fix issue",
        trajectory="Ran reproducer and tests.",
        patch="diff --git a/file.py b/file.py",
        exit_status="Submitted",
        client=client,
        criteria=SWE_CRITERIA[:3],
        n_reps=2,
        model_name="fake",
    )

    assert result.label == VerificationLabel.UNCERTAIN
    assert result.score == 0.5
    assert len(client.prompts) == 6
    assert set(result.criteria_scores) == {"root_cause", "code_review", "empirical_verification"}
