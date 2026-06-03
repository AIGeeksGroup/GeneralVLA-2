from __future__ import annotations

import re


STOPWORDS = {
    "about",
    "after",
    "all",
    "allows",
    "also",
    "and",
    "any",
    "are",
    "before",
    "being",
    "better",
    "but",
    "can",
    "changes",
    "choose",
    "code",
    "complex",
    "correct",
    "dedicated",
    "details",
    "directly",
    "entire",
    "especially",
    "exact",
    "examples",
    "file",
    "first",
    "focus",
    "for",
    "from",
    "helps",
    "identify",
    "in",
    "into",
    "issue",
    "its",
    "it",
    "like",
    "more",
    "most",
    "multi",
    "need",
    "new",
    "of",
    "often",
    "only",
    "or",
    "partial",
    "prefer",
    "reliable",
    "right",
    "solution",
    "specific",
    "structure",
    "that",
    "the",
    "their",
    "then",
    "these",
    "they",
    "this",
    "through",
    "to",
    "tool",
    "tools",
    "use",
    "using",
    "utility",
    "utilities",
    "when",
    "whole",
    "with",
    "write",
}

TOKEN_MAP = {
    "blocks": "block",
    "brittle": "fragile",
    "bug": "failure",
    "bugs": "failure",
    "changed": "edit",
    "changes": "edit",
    "check": "verify",
    "checked": "verify",
    "checking": "verify",
    "commands": "shell",
    "debugging": "debug",
    "edits": "edit",
    "editing": "edit",
    "errorprone": "fragile",
    "errors": "error",
    "example": "reproduce",
    "examples": "reproduce",
    "failing": "failure",
    "fixes": "fix",
    "grep": "search",
    "grep-r": "search",
    "identifies": "locate",
    "identify": "locate",
    "imports": "import",
    "inspecting": "inspect",
    "inspects": "inspect",
    "iterative": "repeat",
    "iteratively": "repeat",
    "lines": "multiline",
    "location": "locate",
    "locations": "locate",
    "locating": "locate",
    "modified": "edit",
    "modification": "edit",
    "modifications": "edit",
    "modify": "edit",
    "patching": "patch",
    "pinpoint": "locate",
    "point": "locate",
    "points": "locate",
    "programmatic": "script",
    "pytest": "test",
    "reads": "inspect",
    "regression": "test",
    "reliably": "verify",
    "replaced": "replace",
    "replaces": "replace",
    "replacing": "replace",
    "reproduction": "reproduce",
    "reproduce": "reproduce",
    "reproducing": "reproduce",
    "rg": "search",
    "scripted": "script",
    "scripts": "script",
    "searching": "search",
    "shell": "shell",
    "snippet": "locate",
    "snippets": "locate",
    "syntaxerrors": "error",
    "targeted": "target",
    "tests": "test",
    "tracing": "trace",
    "understand": "inspect",
    "validated": "verify",
    "verifying": "verify",
}

PHRASE_MAP = {
    "dimensionless inputs": "dimensionless",
    "failing regression test": "failure test",
    "failing test": "failure test",
    "multi line": "multiline",
    "multi-line": "multiline",
    "programmatic solution": "script",
    "programmatic tools": "script",
    "provided examples": "reproduce",
    "python script": "script",
    "reproduction script": "reproduce script",
    "shell commands": "shell",
    "shell utilities": "shell",
    "text manipulation tools": "shell",
}

SIGNATURE_PRIORITY = [
    "reproduce",
    "test",
    "inspect",
    "trace",
    "search",
    "locate",
    "script",
    "sed",
    "edit",
    "patch",
    "verify",
    "debug",
    "multiline",
    "indentation",
    "failure",
    "error",
    "warning",
    "import",
    "dimensionless",
    "conversion",
]

NEGATION_TOKENS = {"avoid", "cannot", "never", "no", "not", "without"}


def _normalize_text(text: str) -> str:
    normalized = text.lower().replace("`", " ")
    for source, target in PHRASE_MAP.items():
        normalized = normalized.replace(source, target)
    normalized = normalized.replace("do not", "dont")
    normalized = normalized.replace("don't", "dont")
    normalized = re.sub(r"[^a-z0-9_+\-\s]", " ", normalized)
    normalized = normalized.replace("-", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_token(token: str) -> str | None:
    token = TOKEN_MAP.get(token, token)
    if token.endswith("ing") and len(token) > 5:
        token = token[:-3]
    elif token.endswith("ed") and len(token) > 4:
        token = token[:-2]
    elif token.endswith("es") and len(token) > 4:
        token = token[:-2]
    elif token.endswith("s") and len(token) > 4:
        token = token[:-1]
    token = TOKEN_MAP.get(token, token)
    if len(token) < 3 or token in STOPWORDS:
        return None
    return token


def canonical_tokens(text: str) -> list[str]:
    normalized = _normalize_text(text)
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[a-z0-9_]{3,}", normalized):
        token = _normalize_token(raw)
        if token is None or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def canonical_token_set(text: str) -> set[str]:
    return set(canonical_tokens(text))


def has_negation(text: str) -> bool:
    normalized = _normalize_text(text)
    words = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    return "dont" in words or bool(words & NEGATION_TOKENS)


def build_signature(text: str, *, max_tokens: int = 6) -> str:
    token_set = canonical_token_set(text)
    prioritized = [token for token in SIGNATURE_PRIORITY if token in token_set]
    remaining = sorted(token for token in token_set if token not in prioritized)
    selected = prioritized if prioritized else remaining
    return "-".join(selected[:max_tokens]) if selected else "memory"


def signature_similarity(left: str, right: str) -> float:
    left_tokens = canonical_token_set(left)
    right_tokens = canonical_token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    ordered_left = build_signature(left, max_tokens=6).split("-")
    ordered_right = build_signature(right, max_tokens=6).split("-")
    signature_overlap = len(set(ordered_left) & set(ordered_right)) / max(len(set(ordered_left) | set(ordered_right)), 1)
    return max(overlap, signature_overlap)
