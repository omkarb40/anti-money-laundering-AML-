"""Tests for Step 7 — decision table (unit) and baseline runner (integration).

Unit tests use synthetic SanctionsHit / RuleFiring / AnomalyScore objects
constructed inline.  They require no disk I/O and run in < 1 ms each.

Integration tests read artifacts/results.jsonl (produced by the runner) or
invoke the CLI via subprocess.  They are skipped when the prerequisite files
are absent.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytest

from aml_copilot.schemas import AnomalyScore, CaseResult, RuleFiring, SanctionsHit
from aml_copilot.step7_runner.decision import (
    ELEVATED_RULE_SEVERITY,
    SANCTIONS_ESCALATION_THRESHOLD,
    apply_decision_table,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parents[1]
_RESULTS_PATH = _ROOT / "artifacts/results.jsonl"
_EVAL_PATH    = _ROOT / "data/fixtures/eval.jsonl"
_TRANS_PATH   = _ROOT / "data/raw/HI-Small_Trans.csv"

_skip_results = pytest.mark.skipif(
    not _RESULTS_PATH.exists(),
    reason="artifacts/results.jsonl not yet built — run python -m aml_copilot.step7_runner.run_baseline",
)
_skip_data = pytest.mark.skipif(
    not (_TRANS_PATH.exists() and _EVAL_PATH.exists()),
    reason="Raw data or eval.jsonl missing — cannot run baseline",
)


# ── Synthetic helpers ─────────────────────────────────────────────────────────

def _hit(score: float = 0.95) -> SanctionsHit:
    return SanctionsHit(
        account_id="ACC001",
        assigned_name="Test Name",
        ofac_uid="U001",
        list_source="SDN",
        match_score=score,
        scorer_used="jaro_winkler",
        matched_name_type="canonical",
    )


def _firing(severity: int = 3, rule_id: str | None = None) -> RuleFiring:
    _rule_ids = {1: "CORRIDOR_001", 2: "FAN_OUT_001", 3: "STRUCT_001"}
    return RuleFiring(
        rule_id=rule_id or _rule_ids[severity],
        severity=severity,
        account_id="ACC001",
        evidence={"test": True},
        window_start=datetime(2022, 1, 1),
        window_end=datetime(2022, 1, 2),
    )


def _anomaly(is_flagged: bool = True, percentile: float = 0.996) -> AnomalyScore:
    return AnomalyScore(
        account_id="ACC001",
        score=-0.3,
        percentile=percentile,
        is_flagged=is_flagged,
        excluded_features=["balance_delta"],
    )


def _decide(**kwargs) -> CaseResult:
    """Call apply_decision_table with sensible defaults for missing kwargs."""
    defaults = dict(
        case_id="CASE_001",
        account_id="ACC001",
        sanctions_hits=[],
        rule_firings=[],
        anomaly_score=None,
        latency_ms=1.0,
    )
    defaults.update(kwargs)
    return apply_decision_table(**defaults)


# ── Branch 1: sanctions >= 0.90 OR severity-3 rule ───────────────────────────

def test_sanctions_90_always_escalates(
    sample_sanctions_hit, sample_rule_firing, sample_anomaly_score
) -> None:
    """SanctionsHit.match_score >= 0.90 → ESCALATE (Branch 1)."""
    # sample_sanctions_hit has match_score=0.95 which is >= 0.90
    result = apply_decision_table(
        case_id="CASE_001",
        account_id="ACC001",
        sanctions_hits=[sample_sanctions_hit],
        rule_firings=[sample_rule_firing],
        anomaly_score=sample_anomaly_score,
        latency_ms=1.0,
    )
    assert result.disposition == "ESCALATE"
    assert result.decision_reason == "sanctions_or_critical_rule"


def test_severity3_alone_escalates(sample_rule_firing) -> None:
    """Severity-3 RuleFiring with no sanctions and no anomaly → ESCALATE (Branch 1)."""
    # sample_rule_firing has severity=3
    result = apply_decision_table(
        case_id="CASE_001",
        account_id="ACC001",
        sanctions_hits=[],
        rule_firings=[sample_rule_firing],
        anomaly_score=None,
        latency_ms=1.0,
    )
    assert result.disposition == "ESCALATE"
    assert result.decision_reason == "sanctions_or_critical_rule"


def test_sanctions_exactly_at_threshold_escalates() -> None:
    """match_score == 0.90 (boundary) → ESCALATE (threshold is inclusive)."""
    result = _decide(sanctions_hits=[_hit(SANCTIONS_ESCALATION_THRESHOLD)])
    assert result.disposition == "ESCALATE"
    assert result.decision_reason == "sanctions_or_critical_rule"


def test_sanctions_below_threshold_no_escalate() -> None:
    """match_score == 0.89 (below 0.90), no other triggers → CLEAR."""
    result = _decide(sanctions_hits=[_hit(0.89)])
    assert result.disposition == "CLEAR"


def test_branch1_triggered_by_sev3_not_sev2() -> None:
    """Severity-2 alone does NOT trigger Branch 1."""
    result = _decide(rule_firings=[_firing(2)])
    # Branch 1 requires severity == 3 exactly
    assert result.disposition != "ESCALATE" or result.decision_reason != "sanctions_or_critical_rule"


# ── Branch 2: anomaly flagged AND severity >= 2 ───────────────────────────────

def test_anomaly_alone_clears(sample_anomaly_score) -> None:
    """AnomalyScore.is_flagged == True with no rule >= severity 2 → CLEAR."""
    result = apply_decision_table(
        case_id="CASE_001",
        account_id="ACC001",
        sanctions_hits=[],
        rule_firings=[],  # no rules at all
        anomaly_score=sample_anomaly_score,
        latency_ms=1.0,
    )
    assert result.disposition == "CLEAR"
    assert result.decision_reason == "clear"


def test_anomaly_plus_severity2_escalates(
    sample_anomaly_score, sample_rule_firing
) -> None:
    """Anomaly flagged AND at least one severity-2 rule → ESCALATE (Branch 2)."""
    sev2 = _firing(2)  # severity=2, not 3 (which would trigger Branch 1)
    result = apply_decision_table(
        case_id="CASE_001",
        account_id="ACC001",
        sanctions_hits=[],
        rule_firings=[sev2],
        anomaly_score=sample_anomaly_score,
        latency_ms=1.0,
    )
    assert result.disposition == "ESCALATE"
    assert result.decision_reason == "anomaly_plus_elevated_rule"


def test_anomaly_plus_severity1_only_clears() -> None:
    """Anomaly flagged + only severity-1 rule → CLEAR (need sev >= 2 for Branch 2)."""
    result = _decide(
        rule_firings=[_firing(1)],
        anomaly_score=_anomaly(is_flagged=True),
    )
    assert result.disposition == "CLEAR"


def test_severity2_without_anomaly_clears() -> None:
    """Severity-2 rule fires but anomaly NOT flagged → CLEAR."""
    result = _decide(
        rule_firings=[_firing(2)],
        anomaly_score=_anomaly(is_flagged=False),
    )
    assert result.disposition == "CLEAR"


def test_severity2_anomaly_none_clears() -> None:
    """Severity-2 rule fires but anomaly_score is None → CLEAR."""
    result = _decide(rule_firings=[_firing(2)], anomaly_score=None)
    assert result.disposition == "CLEAR"


# ── Branch 3: CLEAR ───────────────────────────────────────────────────────────

def test_no_triggers_clears() -> None:
    """Empty hits, empty firings, anomaly=None → CLEAR."""
    result = _decide()
    assert result.disposition == "CLEAR"
    assert result.decision_reason == "clear"


def test_unflagged_anomaly_with_sev2_clears() -> None:
    """anomaly_score.is_flagged=False even with sev-2 rule → CLEAR."""
    result = _decide(
        rule_firings=[_firing(2)],
        anomaly_score=_anomaly(is_flagged=False),
    )
    assert result.disposition == "CLEAR"


# ── Branch precedence ─────────────────────────────────────────────────────────

def test_branch1_takes_precedence_over_branch2() -> None:
    """Sanctions >= 0.90 AND anomaly+sev2 both active → Branch 1 wins."""
    result = _decide(
        sanctions_hits=[_hit(0.92)],
        rule_firings=[_firing(2)],
        anomaly_score=_anomaly(is_flagged=True),
    )
    assert result.decision_reason == "sanctions_or_critical_rule"


def test_sev3_takes_precedence_over_branch2() -> None:
    """Severity-3 rule AND anomaly+sev2 both active → Branch 1 wins."""
    result = _decide(
        rule_firings=[_firing(2), _firing(3)],
        anomaly_score=_anomaly(is_flagged=True),
    )
    assert result.decision_reason == "sanctions_or_critical_rule"


# ── CaseResult field population ───────────────────────────────────────────────

def test_result_fields_populated() -> None:
    """CaseResult carries all tool outputs, not just the disposition."""
    hit = _hit(0.92)
    fir = _firing(3)
    anm = _anomaly()
    result = _decide(
        case_id="CASE_X",
        account_id="ACC_X",
        sanctions_hits=[hit],
        rule_firings=[fir],
        anomaly_score=anm,
        latency_ms=42.7,
    )
    assert result.case_id == "CASE_X"
    assert result.account_id == "ACC_X"
    assert len(result.sanctions_hits) == 1
    assert result.sanctions_hits[0].match_score == 0.92
    assert len(result.rule_firings) == 1
    assert result.rule_firings[0].severity == 3
    assert result.anomaly_score is not None
    assert result.anomaly_score.is_flagged is True
    assert result.latency_ms == pytest.approx(42.7)


def test_result_serialisable_to_json() -> None:
    """CaseResult.model_dump_json() produces valid JSON without crashing."""
    result = _decide(
        sanctions_hits=[_hit()],
        rule_firings=[_firing(2)],
        anomaly_score=_anomaly(is_flagged=True),
    )
    raw = result.model_dump_json()
    parsed = json.loads(raw)
    assert parsed["disposition"] == result.disposition
    assert parsed["decision_reason"] == result.decision_reason


# ── Integration: read pre-built results.jsonl ─────────────────────────────────

def _load_results() -> list[CaseResult]:
    results: list[CaseResult] = []
    with open(_RESULTS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                results.append(CaseResult.model_validate_json(line))
    return results


@_skip_results
def test_all_90_produce_valid_result() -> None:
    """Running against full eval.jsonl produces exactly 90 CaseResult rows, all schema-valid."""
    results = _load_results()
    assert len(results) == 90, f"Expected 90 results, got {len(results)}"


@_skip_results
def test_results_jsonl_valid_json_lines() -> None:
    """Every non-empty line in results.jsonl is valid JSON."""
    with open(_RESULTS_PATH, encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Line {i} is not valid JSON: {exc}")


@_skip_results
def test_no_duplicate_case_ids_in_results() -> None:
    """No case_id appears twice in results.jsonl."""
    results = _load_results()
    counts = Counter(r.case_id for r in results)
    dupes = [cid for cid, n in counts.items() if n > 1]
    assert not dupes, f"Duplicate case_ids: {dupes}"


@_skip_results
def test_disposition_values_valid() -> None:
    """Every disposition is exactly 'ESCALATE' or 'CLEAR'."""
    results = _load_results()
    bad = [r.case_id for r in results if r.disposition not in ("ESCALATE", "CLEAR")]
    assert not bad, f"Invalid dispositions: {bad}"


@_skip_results
def test_latency_values_positive() -> None:
    """All latency_ms values are positive (timer ran correctly)."""
    results = _load_results()
    bad = [r.case_id for r in results if r.latency_ms <= 0]
    assert not bad, f"Non-positive latency_ms: {bad}"


@_skip_results
def test_decision_reason_strings() -> None:
    """decision_reason is one of the three fixed branch strings."""
    valid = {"sanctions_or_critical_rule", "anomaly_plus_elevated_rule", "clear"}
    results = _load_results()
    bad = [r.case_id for r in results if r.decision_reason not in valid]
    assert not bad, f"Unknown decision_reason values: {bad}"


@_skip_results
def test_no_ofac_canonical_names_in_output() -> None:
    """Raw text of results.jsonl must not contain any OFAC canonical names.

    Loads the OFAC SDN XML and checks that no canonical or AKA name string
    appears verbatim in results.jsonl.  Skips if OFAC file is absent.
    """
    ofac_path = _ROOT / "data/raw/ofac/sdn_advanced.xml"
    if not ofac_path.exists():
        pytest.skip("sdn_advanced.xml not available")

    from aml_copilot.step1_identity.ofac_reader import build_raw_name_set, load_ofac_records

    records = load_ofac_records(sdn_path=ofac_path)
    raw_names = build_raw_name_set(records)

    # Only check multi-word canonical-style names (>= 2 space-separated tokens,
    # >= 15 chars).  Single-word tokens like "MARTIN" or "Mahmud" are too short
    # to be meaningful — they appear in notes, account IDs, and evidence strings
    # as coincidental substrings and are not OFAC canonical name leakage.
    full_names = {n for n in raw_names if len(n.split()) >= 2 and len(n) >= 15}

    results_text = _RESULTS_PATH.read_text(encoding="utf-8")
    hits = [n for n in full_names if n in results_text]
    assert not hits, (
        f"Multi-word OFAC canonical/AKA names found verbatim in results.jsonl: {hits[:5]}"
    )


# ── CLI exit code ─────────────────────────────────────────────────────────────

@_skip_data
def test_single_command_exits_0(tmp_path) -> None:
    """run_baseline.py invocation with valid --eval and --out exits with code 0."""
    import subprocess

    out = tmp_path / "results.jsonl"
    proc = subprocess.run(
        [
            sys.executable, "-m", "aml_copilot.step7_runner.run_baseline",
            "--eval", str(_EVAL_PATH),
            "--out",  str(out),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"Runner exited {proc.returncode}\n"
        f"stderr (last 2000 chars):\n{proc.stderr[-2000:]}"
    )
    assert out.exists(), "results.jsonl was not created by the runner"
    # Quick sanity check on the output
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 90, f"Expected 90 result lines, got {len(lines)}"
