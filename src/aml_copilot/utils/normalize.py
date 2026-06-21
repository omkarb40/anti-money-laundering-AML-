from __future__ import annotations


def unicode_normalize(text: str) -> str:
    """NFKD-normalize and strip combining characters for transliteration matching."""
    ...


def normalize_name(name: str) -> str:
    """Lowercase, unicode-normalize, and strip extra whitespace."""
    ...
