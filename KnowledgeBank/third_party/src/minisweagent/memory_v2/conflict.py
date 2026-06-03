from __future__ import annotations

from .signatures import canonical_token_set, has_negation, signature_similarity

DIRECTIVE_TOKENS = {"debug", "edit", "inspect", "locate", "patch", "reproduce", "search", "test", "trace", "verify"}


def may_conflict(left: str, right: str) -> bool:
    left_tokens = canonical_token_set(left)
    right_tokens = canonical_token_set(right)
    shared_directives = bool((left_tokens & right_tokens) & DIRECTIVE_TOKENS)
    opposite_polarity = has_negation(left) != has_negation(right)
    return opposite_polarity and (shared_directives or signature_similarity(left, right) >= 0.3)
