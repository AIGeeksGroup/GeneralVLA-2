from minisweagent.verifier.schema import VerificationLabel, VerificationResult


def test_verification_result_labels_success_fail_and_uncertain_from_thresholds():
    success = VerificationResult.from_scores(
        criteria_scores={"root_cause": 0.8, "code_review": 0.7, "empirical_verification": 0.75},
        raw_outputs=["ok"],
        model_name="gemini-2.5-flash",
        n_reps=1,
        used_logprobs=False,
    )
    fail = VerificationResult.from_scores(
        criteria_scores={"root_cause": 0.2, "code_review": 0.3, "empirical_verification": 0.35},
        raw_outputs=["bad"],
        model_name="gemini-2.5-flash",
        n_reps=1,
        used_logprobs=False,
    )
    uncertain = VerificationResult.from_scores(
        criteria_scores={"root_cause": 0.5, "code_review": 0.6, "empirical_verification": 0.55},
        raw_outputs=["mixed"],
        model_name="gemini-2.5-flash",
        n_reps=1,
        used_logprobs=False,
    )

    assert success.label == VerificationLabel.VERIFIED_SUCCESS
    assert fail.label == VerificationLabel.VERIFIED_FAIL
    assert uncertain.label == VerificationLabel.UNCERTAIN
    assert round(success.score, 4) == 0.75


def test_verification_result_caps_non_submitted_trajectory_below_success_threshold():
    result = VerificationResult.from_scores(
        criteria_scores={"root_cause": 0.95, "code_review": 0.95, "empirical_verification": 0.95},
        raw_outputs=["looks good"],
        model_name="gemini-2.5-flash",
        n_reps=1,
        used_logprobs=True,
        exit_status="RuntimeError",
    )

    assert result.score == 0.49
    assert result.label == VerificationLabel.UNCERTAIN
