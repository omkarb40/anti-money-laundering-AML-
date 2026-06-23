"""Tests for Step 6 eval set — validates the frozen eval.jsonl artifact.

All tests read directly from data/fixtures/eval.jsonl (the frozen file).
Tests skip gracefully if the file has not yet been built (clean environment
without raw data).  Once the file is frozen every test must pass and must
continue to pass — these are artifact-integrity guards, not builder-logic tests.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from aml_copilot.schemas import EvalCase
from aml_copilot.step6_eval.builder import CONFLICT_TARGETS, SLICE_COUNTS, TYPOLOGY_TARGETS

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parents[1]
_EVAL_PATH = _ROOT / "data/fixtures/eval.jsonl"
_CHECKSUM_PATH = _ROOT / "artifacts/checksums.sha256"

# ── Shared fixture ────────────────────────────────────────────────────────────


def _load_cases() -> list[EvalCase]:
    """Load and parse all EvalCase objects from eval.jsonl."""
    cases: list[EvalCase] = []
    with open(_EVAL_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(EvalCase.model_validate_json(line))
    return cases


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


_skip_if_missing = pytest.mark.skipif(
    not _EVAL_PATH.exists(),
    reason="data/fixtures/eval.jsonl not yet built — run python -m aml_copilot.step6_eval.builder",
)


# ── Core integrity ────────────────────────────────────────────────────────────


@_skip_if_missing
def test_eval_exactly_90_rows() -> None:
    """Built eval set has exactly 90 EvalCase rows."""
    cases = _load_cases()
    assert len(cases) == 90, f"Expected 90 rows, got {len(cases)}"


@_skip_if_missing
def test_eval_schema_valid() -> None:
    """All 90 rows parse as EvalCase without Pydantic ValidationError."""
    # _load_cases() already calls model_validate_json; if it raises, test fails.
    cases = _load_cases()
    assert len(cases) == 90


@_skip_if_missing
def test_eval_checksum_matches() -> None:
    """SHA-256 of eval.jsonl matches the entry recorded in checksums.sha256."""
    assert _CHECKSUM_PATH.exists(), f"Checksum file not found: {_CHECKSUM_PATH}"

    from aml_copilot.utils.checksum import _to_key
    actual = _compute_sha256(_EVAL_PATH)
    eval_key = _to_key(_EVAL_PATH)

    recorded: str | None = None
    for line in _CHECKSUM_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) == 2 and parts[1] == eval_key:
            recorded = parts[0]
            break

    assert recorded is not None, (
        f"No checksum entry found for {_EVAL_PATH} in {_CHECKSUM_PATH}"
    )
    assert actual == recorded, (
        f"Checksum mismatch for eval.jsonl\n"
        f"  recorded: {recorded}\n"
        f"  actual:   {actual}"
    )


@_skip_if_missing
def test_slice_counts() -> None:
    """Each of the 5 case_type values has the expected count per SLICE_COUNTS."""
    cases = _load_cases()
    counts = Counter(c.case_type for c in cases)
    for case_type, expected in SLICE_COUNTS.items():
        assert counts[case_type] == expected, (
            f"case_type={case_type!r}: expected {expected}, got {counts[case_type]}"
        )


# ── Uniqueness constraints ────────────────────────────────────────────────────


@_skip_if_missing
def test_no_duplicate_case_ids() -> None:
    """Every case_id is unique across the 90 cases."""
    cases = _load_cases()
    counts = Counter(c.case_id for c in cases)
    dupes = [cid for cid, n in counts.items() if n > 1]
    assert not dupes, f"Duplicate case_ids: {dupes}"


@_skip_if_missing
def test_no_duplicate_account_ids() -> None:
    """Every account_id appears in at most one eval case (no surrogate accounts)."""
    cases = _load_cases()
    counts = Counter(c.account_id for c in cases)
    dupes = [aid for aid, n in counts.items() if n > 1]
    assert not dupes, f"Duplicate account_ids: {dupes}"


# ── Gold label correctness ────────────────────────────────────────────────────


@_skip_if_missing
def test_sanctions_hit_all_escalate() -> None:
    """All sanctions_hit cases are gold-labeled ESCALATE (Branch 1 of decision table)."""
    cases = _load_cases()
    bad = [c.case_id for c in cases if c.case_type == "sanctions_hit" and c.gold_label != "ESCALATE"]
    assert not bad, f"sanctions_hit cases with non-ESCALATE label: {bad}"


@_skip_if_missing
def test_sanctions_near_miss_all_clear() -> None:
    """All sanctions_near_miss cases are gold-labeled CLEAR (HN below 0.90 threshold)."""
    cases = _load_cases()
    bad = [c.case_id for c in cases if c.case_type == "sanctions_near_miss" and c.gold_label != "CLEAR"]
    assert not bad, f"sanctions_near_miss cases with non-CLEAR label: {bad}"


@_skip_if_missing
def test_ibm_labeled_all_escalate() -> None:
    """All ibm_labeled cases are gold-labeled ESCALATE (IBM ground truth)."""
    cases = _load_cases()
    bad = [c.case_id for c in cases if c.case_type == "ibm_labeled" and c.gold_label != "ESCALATE"]
    assert not bad, f"ibm_labeled cases with non-ESCALATE label: {bad}"


@_skip_if_missing
def test_typology_all_escalate() -> None:
    """All typology cases are gold-labeled ESCALATE (IBM laundering ground truth)."""
    cases = _load_cases()
    bad = [c.case_id for c in cases if c.case_type == "typology" and c.gold_label != "ESCALATE"]
    assert not bad, f"typology cases with non-ESCALATE label: {bad}"


@_skip_if_missing
def test_escalate_count_minimum() -> None:
    """At least 45 of 90 cases are ESCALATE (Step 8 denominator guard)."""
    cases = _load_cases()
    n_escalate = sum(1 for c in cases if c.gold_label == "ESCALATE")
    assert n_escalate >= 45, f"Only {n_escalate} ESCALATE cases; need ≥ 45"


# ── Slice-specific field validation ──────────────────────────────────────────


@_skip_if_missing
def test_typology_field_set_for_typology_cases() -> None:
    """Every typology case has a non-None typology field matching TYPOLOGY_TARGETS."""
    cases = _load_cases()
    valid_typologies = set(TYPOLOGY_TARGETS.keys())
    for c in cases:
        if c.case_type == "typology":
            assert c.typology is not None, f"{c.case_id} missing typology field"
            assert c.typology in valid_typologies, (
                f"{c.case_id} has unknown typology={c.typology!r}"
            )


@_skip_if_missing
def test_typology_counts_per_bucket() -> None:
    """Each typology bucket has the target count from TYPOLOGY_TARGETS."""
    cases = _load_cases()
    typ_counts = Counter(
        c.typology for c in cases if c.case_type == "typology" and c.typology is not None
    )
    for typ, expected in TYPOLOGY_TARGETS.items():
        assert typ_counts[typ] == expected, (
            f"typology={typ!r}: expected {expected}, got {typ_counts[typ]}"
        )


@_skip_if_missing
def test_ibm_labeled_severity_bands() -> None:
    """ibm_labeled cases have exactly 10 cases per severity band (1, 2, 3)."""
    cases = _load_cases()
    band_counts = Counter(
        c.severity_band for c in cases
        if c.case_type == "ibm_labeled" and c.severity_band is not None
    )
    for band in (1, 2, 3):
        assert band_counts[band] == 10, (
            f"severity_band={band}: expected 10, got {band_counts[band]}"
        )


@_skip_if_missing
def test_conflict_subtypes() -> None:
    """Conflict cases have the correct subtype distribution from CONFLICT_TARGETS."""
    cases = _load_cases()
    subtype_counts = Counter(
        c.conflict_type for c in cases
        if c.case_type == "rules_anomaly_conflict" and c.conflict_type is not None
    )
    for subtype, expected in CONFLICT_TARGETS.items():
        assert subtype_counts[subtype] == expected, (
            f"conflict_type={subtype!r}: expected {expected}, got {subtype_counts[subtype]}"
        )


@_skip_if_missing
def test_conflict_rule3_escalates() -> None:
    """rule3_no_anomaly conflict cases are ESCALATE (severity-3 rule fires → Branch 1)."""
    cases = _load_cases()
    bad = [
        c.case_id for c in cases
        if c.case_type == "rules_anomaly_conflict"
        and c.conflict_type == "rule3_no_anomaly"
        and c.gold_label != "ESCALATE"
    ]
    assert not bad, f"rule3_no_anomaly cases with wrong label: {bad}"


@_skip_if_missing
def test_conflict_clears_are_clear() -> None:
    """anomaly_no_rule and rule_no_anomaly conflict cases are CLEAR (decision table Branch 3)."""
    cases = _load_cases()
    bad = [
        c.case_id for c in cases
        if c.case_type == "rules_anomaly_conflict"
        and c.conflict_type in ("anomaly_no_rule", "rule_no_anomaly")
        and c.gold_label != "CLEAR"
    ]
    assert not bad, f"Should-be-CLEAR conflict cases with wrong label: {bad}"


# ── File format integrity ─────────────────────────────────────────────────────


@_skip_if_missing
def test_eval_jsonl_valid_json_lines() -> None:
    """Every non-empty line in eval.jsonl is valid JSON."""
    with open(_EVAL_PATH, encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Line {i} is not valid JSON: {exc}")


@_skip_if_missing
def test_no_synthetic_surrogate_accounts() -> None:
    """No case_id uses a placeholder/surrogate prefix that would indicate a synthetic account."""
    cases = _load_cases()
    surrogate_prefixes = ("SYNTH_", "FAKE_", "PLACEHOLDER_", "DUMMY_")
    bad = [c.case_id for c in cases if any(c.account_id.startswith(p) for p in surrogate_prefixes)]
    assert not bad, f"Surrogate account_ids found: {bad}"
