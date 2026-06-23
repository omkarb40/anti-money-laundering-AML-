"""Shared Unicode normalization and name-scoring utilities."""
from __future__ import annotations

import unicodedata


def nfkd_normalize(text: str) -> str:
    """NFKD decompose then strip combining characters (é→e, ü→u, ñ→n)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def normalize_name(name: str) -> str:
    """Lowercase + NFKD + collapse whitespace. Used as the canonical form for scoring."""
    return " ".join(nfkd_normalize(name).lower().split())


def score_names(query: str, candidate: str) -> float:
    """
    Step 2-compatible scoring: normalize both sides, return max(JW, token_sort_ratio).
    Imported here so Step 1 ground-truth construction uses identical logic to Step 2.
    """
    from rapidfuzz.distance import JaroWinkler
    from rapidfuzz import fuzz

    q = normalize_name(query)
    c = normalize_name(candidate)
    jw = JaroWinkler.similarity(q, c)
    tsr = fuzz.token_sort_ratio(q, c) / 100.0
    return max(jw, tsr)
