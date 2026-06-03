from minisweagent.verifier.scales import (
    SCORE_TAG,
    extract_tagged_score_token,
    normalize_score_token,
    score_from_token_distribution,
)


def test_normalize_score_token_maps_a_to_best_and_t_to_worst():
    assert normalize_score_token("A") == 1.0
    assert normalize_score_token("T") == 0.0
    assert normalize_score_token("a") == 1.0


def test_extract_tagged_score_token_reads_score_tag_case_insensitively():
    text = "Analysis here.\n<score> c </score>"
    assert extract_tagged_score_token(text, SCORE_TAG) == "C"


def test_score_from_token_distribution_uses_probability_weighted_expected_value():
    score = score_from_token_distribution([("A", 0.25), ("T", 0.75)])
    assert round(score, 4) == 0.25


def test_score_from_token_distribution_falls_back_to_midpoint_without_valid_tokens():
    assert score_from_token_distribution([("?", 1.0)]) == 0.5
