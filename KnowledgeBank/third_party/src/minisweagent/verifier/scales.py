from __future__ import annotations

import math
import re


GRANULARITY = 20
SCORE_TAG = "score"

_SCORE_VALUES = {chr(65 + index): float(GRANULARITY - index) for index in range(GRANULARITY)}

SCALE_DESCRIPTION = (
    "Rate how likely the agent correctly solved the task on a 20-point scale using letters A through T:\n"
    "  A = clearly and completely succeeded with verified output (best)\n"
    "  B-D = succeeded with only minor issues\n"
    "  E-G = above average, mostly correct with some issues\n"
    "  H-J = uncertain, leans toward success\n"
    "  K-M = uncertain, leans toward failure\n"
    "  N-P = below average, significant issues remain\n"
    "  Q-S = failed with some partial progress\n"
    "  T = clearly and completely failed (worst)"
)


def normalize_score_token(token: str) -> float | None:
    raw_value = _SCORE_VALUES.get(token.strip().upper())
    if raw_value is None:
        return None
    return (raw_value - 1.0) / float(GRANULARITY - 1)


def extract_tagged_score_token(text: str, tag: str = SCORE_TAG) -> str | None:
    pattern = rf"<{re.escape(tag)}>\s*([A-Ta-t])\s*</{re.escape(tag)}>"
    match = re.search(pattern, text or "", re.IGNORECASE)
    return match.group(1).strip().upper() if match else None


def score_from_text(text: str, tag: str = SCORE_TAG) -> float:
    token = extract_tagged_score_token(text, tag)
    if token is None:
        return 0.5
    score = normalize_score_token(token)
    return 0.5 if score is None else score


def score_from_token_distribution(token_probs: list[tuple[str, float]]) -> float:
    weighted = 0.0
    total = 0.0
    for token, probability in token_probs:
        score = normalize_score_token(token)
        if score is None:
            continue
        weighted += score * probability
        total += probability
    if total <= 0:
        return 0.5
    return weighted / total


def score_from_logprob_distribution(token_logprobs: list[tuple[str, float]]) -> float:
    return score_from_token_distribution([(token, math.exp(logprob)) for token, logprob in token_logprobs])
