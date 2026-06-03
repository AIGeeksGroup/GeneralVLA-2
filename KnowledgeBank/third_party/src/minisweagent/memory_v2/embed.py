from __future__ import annotations

from typing import Callable


def build_query_embedding(query: str, embed_fn: Callable[[str], list[float]]) -> list[float]:
    """Compute a transient query embedding without mutating persistent cache."""
    return embed_fn(query)
