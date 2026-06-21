from __future__ import annotations

from aml_copilot.schemas import SanctionsHit
from aml_copilot.step2_sanctions.index import OFACEntry

MATCH_THRESHOLD: float = 0.85      # minimum score to return a hit
ESCALATION_THRESHOLD: float = 0.90 # score at which decision table escalates


def screen_name(
    name: str,
    ofac_index: dict[str, OFACEntry],
    account_id: str,
) -> list[SanctionsHit]:
    """
    Return all SanctionsHit objects with score >= MATCH_THRESHOLD.
    Pipeline: unicode_normalize → JW score → token_sort_ratio score → max(both).
    """
    ...


def compute_score(name_a: str, name_b: str) -> float:
    """Return max(jaro_winkler, token_sort_ratio/100) for two pre-normalized strings."""
    ...
