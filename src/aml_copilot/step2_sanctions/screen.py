"""
Step 2: OFAC sanctions screening.

screen_account() applies Jaro-Winkler + token_sort_ratio to match a single
account name against the pre-built OFACIndex. Both scorers are always
computed; the higher one is used. Reordered names rely on token_sort_ratio
and must never be pre-filtered.

assigned_name is carried in SanctionsHit for internal traceability only.
Callers must not print or log it at INFO level or above.
"""
from __future__ import annotations

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from aml_copilot.schemas import SanctionsHit
from aml_copilot.step2_sanctions.index import OFACIndex
from aml_copilot.utils.normalize import normalize_name

MATCH_THRESHOLD: float = 0.85       # minimum score to include in results
ESCALATION_THRESHOLD: float = 0.90  # Step 7 decision gate (read-only here)


def compute_score(norm_a: str, norm_b: str) -> tuple[float, str]:
    """
    Return (score, scorer_used) for two pre-normalized name strings.

    Both JW and TSR are always computed. scorer_used is "jaro_winkler" when
    JW >= TSR, "token_sort_ratio" otherwise.  Callers must pre-normalize both
    arguments with normalize_name() before calling.
    """
    jw = JaroWinkler.similarity(norm_a, norm_b)
    tsr = fuzz.token_sort_ratio(norm_a, norm_b) / 100.0
    if jw >= tsr:
        return jw, "jaro_winkler"
    return tsr, "token_sort_ratio"


def screen_account(
    account_id: str,
    name: str,
    index: OFACIndex,
) -> list[SanctionsHit]:
    """
    Screen a single account name against the full OFAC index.

    Returns all SanctionsHit with match_score >= MATCH_THRESHOLD, sorted
    descending by match_score.  Results are deduplicated by ofac_uid: when
    the same uid matches via both a canonical name and an AKA, only the
    higher-scoring variant is returned.

    Parameters
    ----------
    account_id : str
        Account identifier — included in every SanctionsHit for traceability.
    name : str
        Account's assigned name from identity_overlay.  Carried internally in
        SanctionsHit.assigned_name but must not be logged by callers.
    index : OFACIndex
        Pre-built index from build_ofac_index().
    """
    norm_input = normalize_name(name)
    if not norm_input:
        return []

    # hits_by_uid tracks the best hit seen so far per OFAC entity
    hits_by_uid: dict[str, SanctionsHit] = {}

    def _update(uid: str, score: float, scorer: str, entry_is_canonical: bool, list_name: str) -> None:
        existing = hits_by_uid.get(uid)
        if existing is not None and score <= existing.match_score:
            return
        hits_by_uid[uid] = SanctionsHit(
            account_id=account_id,
            assigned_name=name,
            ofac_uid=uid,
            list_source=list_name,
            match_score=score,
            scorer_used=scorer,
            matched_name_type="canonical" if entry_is_canonical else "alias",
        )

    # Phase 1: O(1) exact-match fast path
    for entry in index.exact_map.get(norm_input, []):
        _update(entry.uid, 1.0, "exact", entry.is_canonical, entry.list_name)

    # Phase 2: fuzzy scan over all entries — exact matches already in hits_by_uid
    # at score=1.0, so the <= guard prevents downgrading them
    for entry in index.all_entries:
        score, scorer = compute_score(norm_input, entry.normalized_name)
        if score < MATCH_THRESHOLD:
            continue
        _update(entry.uid, score, scorer, entry.is_canonical, entry.list_name)

    return sorted(hits_by_uid.values(), key=lambda h: h.match_score, reverse=True)
