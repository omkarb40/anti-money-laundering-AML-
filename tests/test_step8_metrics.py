"""Tests for Step 8 — metrics computation (unit) and frozen artifact (integration).

Unit tests build synthetic EvalCase / CaseResult objects inline.
They have no disk I/O and run in < 1 ms each.

Integration tests read artifacts/metrics_baseline.json once it is produced by
the CLI and are skipped if the file is absent.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aml_copilot.schemas import CaseResult, EvalCase, MetricsReport
from aml_copilot.step8_metrics.metrics import (
    SEVERITY_WEIGHTS,
    _weight,
    compute_metrics,
    save_metrics,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT         = Path(__file__).parents[1]
_METRICS_PATH = _ROOT / "artifacts/metrics_baseline.json"
_RESULTS_PATH = _ROOT / "artifacts/results.jsonl"
_EVAL_PATH    = _ROOT / "data/fixtures/eval.jsonl"

_skip_metrics = pytest.mark.skipif(
    not _METRICS_PATH.exists(),
    reason="artifacts/metrics_baseline.json not yet built — run python -m aml_copilot.step8_metrics.metrics",
)
_skip_inputs = pytest.mark.skipif(
    not (_RESULTS_PATH.exists() and _EVAL_PATH.exists()),
    reason="results.jsonl or eval.jsonl missing",
)


# ── Synthetic helpers ─────────────────────────────────────────────────────────

def _eval(
    case_id: str,
    gold_label: str,
    case_type: str = "ibm_labeled",
    severity_band: int | None = None,
    conflict_type: str | None = None,
) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        account_id=f"ACC{case_id}",
        gold_label=gold_label,
        case_type=case_type,
        severity_band=severity_band,
        conflict_type=conflict_type,
        relevant_txn_ids=[],
        notes="unit-test case",
    )


def _result(case_id: str, disposition: str, latency_ms: float = 10.0) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        account_id=f"ACC{case_id}",
        disposition=disposition,
        decision_reason="test",
        sanctions_hits=[],
        rule_firings=[],
        anomaly_score=None,
        latency_ms=latency_ms,
    )


def _minimal_report() -> MetricsReport:
    """One-pair MetricsReport for write/freeze tests."""
    e = [_eval("X0", "ESCALATE", severity_band=1)]
    r = [_result("X0", "ESCALATE")]
    return compute_metrics(r, e)


# ── Unit: accuracy ────────────────────────────────────────────────────────────

def test_perfect_results_accuracy_1() -> None:
    """If all dispositions match gold labels, disposition_accuracy == 1.0."""
    evals = (
        [_eval(f"E{i}", "ESCALATE", severity_band=2) for i in range(5)]
        + [_eval(f"C{i}", "CLEAR") for i in range(5)]
    )
    results = (
        [_result(f"E{i}", "ESCALATE") for i in range(5)]
        + [_result(f"C{i}", "CLEAR") for i in range(5)]
    )
    report = compute_metrics(results, evals)
    assert report.disposition_accuracy == pytest.approx(1.0)


def test_all_wrong_accuracy_0() -> None:
    """If every disposition is inverted, disposition_accuracy == 0.0."""
    evals   = [_eval("A", "ESCALATE", severity_band=1), _eval("B", "CLEAR")]
    results = [_result("A", "CLEAR"), _result("B", "ESCALATE")]
    report  = compute_metrics(results, evals)
    assert report.disposition_accuracy == pytest.approx(0.0)


def test_accuracy_partial() -> None:
    """Two correct out of three → accuracy == 2/3."""
    evals   = [_eval("A", "ESCALATE", severity_band=1), _eval("B", "ESCALATE", severity_band=1), _eval("C", "CLEAR")]
    results = [_result("A", "ESCALATE"), _result("B", "CLEAR"), _result("C", "CLEAR")]
    report  = compute_metrics(results, evals)
    assert report.disposition_accuracy == pytest.approx(2 / 3)


# ── Unit: weighted false-clear rate ──────────────────────────────────────────

def test_false_clear_rate_weighting() -> None:
    """A severity-3 false clear contributes 3× the weight of a severity-1 false clear."""
    e_sev3 = _eval("A", "ESCALATE", severity_band=3)
    e_sev1 = _eval("B", "ESCALATE", severity_band=1)
    eval_cases = [e_sev3, e_sev1]

    # Scenario A: sev-3 is FN (missed), sev-1 is TP (caught)
    report_a = compute_metrics(
        [_result("A", "CLEAR"), _result("B", "ESCALATE")],
        eval_cases,
    )
    # Scenario B: sev-3 is TP (caught), sev-1 is FN (missed)
    report_b = compute_metrics(
        [_result("A", "ESCALATE"), _result("B", "CLEAR")],
        eval_cases,
    )
    # weighted_denom same in both (4) → ratio of FCR == ratio of weighted FN
    # FCR_A = 3/4 = 0.75,  FCR_B = 1/4 = 0.25  → ratio = 3
    assert report_a.false_clear_rate_weighted == pytest.approx(
        3.0 * report_b.false_clear_rate_weighted
    )


def test_fcr_zero_when_all_escalate_caught() -> None:
    """No false clears → FCR_weighted == 0.0."""
    evals   = [_eval(f"A{i}", "ESCALATE", severity_band=3) for i in range(3)]
    results = [_result(f"A{i}", "ESCALATE") for i in range(3)]
    report  = compute_metrics(results, evals)
    assert report.false_clear_rate_weighted == pytest.approx(0.0)


def test_fcr_1_when_all_escalate_missed() -> None:
    """All ESCALATE gold cases predicted CLEAR → FCR_weighted == 1.0."""
    evals   = [_eval(f"A{i}", "ESCALATE", severity_band=1) for i in range(4)]
    results = [_result(f"A{i}", "CLEAR") for i in range(4)]
    report  = compute_metrics(results, evals)
    assert report.false_clear_rate_weighted == pytest.approx(1.0)


def test_weight_function_case_types() -> None:
    """_weight() returns the correct value for each case type / subtype."""
    assert _weight(_eval("x", "ESCALATE", "ibm_labeled", severity_band=1)) == pytest.approx(1.0)
    assert _weight(_eval("x", "ESCALATE", "ibm_labeled", severity_band=2)) == pytest.approx(2.0)
    assert _weight(_eval("x", "ESCALATE", "ibm_labeled", severity_band=3)) == pytest.approx(3.0)
    assert _weight(_eval("x", "ESCALATE", "sanctions_hit"))                 == pytest.approx(3.0)
    assert _weight(_eval("x", "ESCALATE", "typology"))                      == pytest.approx(2.0)
    assert _weight(_eval("x", "ESCALATE", "rules_anomaly_conflict", conflict_type="rule3_no_anomaly")) == pytest.approx(3.0)
    assert _weight(_eval("x", "ESCALATE", "rules_anomaly_conflict", conflict_type="rule_no_anomaly"))  == pytest.approx(1.0)


# ── Unit: sanctions precision / recall ───────────────────────────────────────

def test_sanctions_precision_recall_perfect() -> None:
    """All sanctions_hit → ESCALATE and all near_miss → CLEAR → both == 1.0."""
    evals = (
        [_eval(f"SH{i}", "ESCALATE", "sanctions_hit")      for i in range(3)]
        + [_eval(f"SNM{i}", "CLEAR", "sanctions_near_miss") for i in range(3)]
    )
    results = (
        [_result(f"SH{i}", "ESCALATE")  for i in range(3)]
        + [_result(f"SNM{i}", "CLEAR")  for i in range(3)]
    )
    report = compute_metrics(results, evals)
    assert report.sanctions_precision == pytest.approx(1.0)
    assert report.sanctions_recall    == pytest.approx(1.0)


def test_sanctions_fp_reduces_precision() -> None:
    """One near_miss escalated → precision < 1.0."""
    evals = (
        [_eval("SH0", "ESCALATE", "sanctions_hit")]
        + [_eval("SNM0", "CLEAR", "sanctions_near_miss")]   # hard negative
    )
    results = [
        _result("SH0",  "ESCALATE"),   # TP
        _result("SNM0", "ESCALATE"),   # FP — near-miss escalated
    ]
    report = compute_metrics(results, evals)
    # precision = TP / (TP + FP) = 1 / 2
    assert report.sanctions_precision == pytest.approx(0.5)
    assert report.sanctions_recall    == pytest.approx(1.0)


def test_sanctions_fn_reduces_recall() -> None:
    """One sanctions_hit cleared → recall < 1.0."""
    evals = (
        [_eval("SH0", "ESCALATE", "sanctions_hit")]
        + [_eval("SH1", "ESCALATE", "sanctions_hit")]
    )
    results = [
        _result("SH0", "ESCALATE"),   # TP
        _result("SH1", "CLEAR"),       # FN — hit missed
    ]
    report = compute_metrics(results, evals)
    # recall = TP / (TP + FN) = 1 / 2
    assert report.sanctions_recall    == pytest.approx(0.5)
    assert report.sanctions_precision == pytest.approx(1.0)  # no FP


def test_sanctions_zero_safe() -> None:
    """No sanctions cases in eval → precision == recall == 0.0, no ZeroDivisionError."""
    evals   = [_eval("A", "ESCALATE", "ibm_labeled", severity_band=1)]
    results = [_result("A", "ESCALATE")]
    report  = compute_metrics(results, evals)
    assert report.sanctions_precision == pytest.approx(0.0)
    assert report.sanctions_recall    == pytest.approx(0.0)


# ── Unit: denominator safety ──────────────────────────────────────────────────

def test_denominator_zero_safe() -> None:
    """Empty results list raises ValueError, not ZeroDivisionError."""
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([], [_eval("A", "ESCALATE")])


def test_empty_eval_raises_valueerror() -> None:
    """Empty eval_cases list raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([_result("A", "ESCALATE")], [])


def test_all_clear_gold_fcr_zero() -> None:
    """Eval with no ESCALATE cases → FCR_weighted == 0.0 without ZeroDivisionError."""
    evals   = [_eval("C0", "CLEAR"), _eval("C1", "CLEAR")]
    results = [_result("C0", "CLEAR"), _result("C1", "CLEAR")]
    report  = compute_metrics(results, evals)          # must not raise
    assert report.false_clear_rate_weighted == pytest.approx(0.0)


# ── Unit: join validation ─────────────────────────────────────────────────────

def test_join_missing_result_raises() -> None:
    """Eval has case_id 'X'; results has only 'Y' → ValueError naming the missing id."""
    with pytest.raises(ValueError, match="missing"):
        compute_metrics([_result("Y", "ESCALATE")], [_eval("X", "ESCALATE", severity_band=1)])


def test_join_extra_result_raises() -> None:
    """Results has case_id 'Y' not in eval → ValueError naming the extra id."""
    evals   = [_eval("A", "ESCALATE", severity_band=1)]
    results = [_result("A", "ESCALATE"), _result("Y", "CLEAR")]
    with pytest.raises(ValueError, match="not in eval"):
        compute_metrics(results, evals)


def test_join_duplicate_result_raises() -> None:
    """Two results with the same case_id → ValueError."""
    evals   = [_eval("A", "ESCALATE", severity_band=1)]
    results = [_result("A", "ESCALATE"), _result("A", "CLEAR")]
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        compute_metrics(results, evals)


# ── Unit: fixed fields ────────────────────────────────────────────────────────

def test_total_cost_is_zero() -> None:
    """total_cost_usd is always 0.0 (no LLM calls in Phase 1–3)."""
    report = compute_metrics([_result("A", "ESCALATE")], [_eval("A", "ESCALATE", severity_band=1)])
    assert report.total_cost_usd == 0.0


def test_eval_size_matches_input() -> None:
    """eval_size == number of eval cases passed."""
    n = 7
    evals   = [_eval(f"C{i}", "CLEAR") for i in range(n)]
    results = [_result(f"C{i}", "CLEAR") for i in range(n)]
    report  = compute_metrics(results, evals)
    assert report.eval_size == n


def test_generated_at_is_utc_aware() -> None:
    """generated_at must carry UTC timezone info."""
    report = compute_metrics([_result("A", "CLEAR")], [_eval("A", "CLEAR")])
    assert report.generated_at.tzinfo is not None
    assert report.generated_at.utcoffset().total_seconds() == 0  # type: ignore[union-attr]


# ── Unit: latency percentiles ─────────────────────────────────────────────────

def test_latency_percentiles_correct() -> None:
    """p50 and p95 match numpy.percentile of the latency_ms values."""
    import numpy as np

    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    evals   = [_eval(f"L{i}", "CLEAR") for i in range(10)]
    results = [_result(f"L{i}", "CLEAR", latency_ms=lat) for i, lat in enumerate(latencies)]
    report  = compute_metrics(results, evals)
    assert report.latency_p50_ms == pytest.approx(np.percentile(latencies, 50))
    assert report.latency_p95_ms == pytest.approx(np.percentile(latencies, 95))


# ── Unit: write-once freeze ───────────────────────────────────────────────────

def test_metrics_frozen_after_write(tmp_path) -> None:
    """Writing metrics_baseline.json records SHA-256; second write raises RuntimeError."""
    report         = _minimal_report()
    metrics_path   = tmp_path / "metrics_baseline.json"
    checksums_path = tmp_path / "checksums.sha256"

    # First write: must succeed and return a hex digest
    digest = save_metrics(report, path=metrics_path, checksums_path=checksums_path)
    assert metrics_path.exists()
    assert len(digest) == 64  # SHA-256 hex string

    # Second write without force: must raise RuntimeError (write-once)
    with pytest.raises(RuntimeError):
        save_metrics(report, path=metrics_path, checksums_path=checksums_path)

    # Second write with force=True: must succeed
    save_metrics(report, path=metrics_path, checksums_path=checksums_path, force=True)
    assert metrics_path.exists()


def test_save_metrics_produces_valid_json(tmp_path) -> None:
    """Written metrics_baseline.json parses as valid MetricsReport."""
    report         = _minimal_report()
    metrics_path   = tmp_path / "metrics_baseline.json"
    checksums_path = tmp_path / "checksums.sha256"

    save_metrics(report, path=metrics_path, checksums_path=checksums_path)
    loaded = MetricsReport.model_validate_json(metrics_path.read_text(encoding="utf-8"))
    assert loaded.total_cost_usd == 0.0
    assert loaded.eval_size == 1


def test_save_metrics_nonzero_cost_raises(tmp_path) -> None:
    """save_metrics raises AssertionError if total_cost_usd != 0.0."""
    from datetime import timezone
    bad_report = MetricsReport(
        disposition_accuracy=1.0,
        false_clear_rate_weighted=0.0,
        sanctions_precision=1.0,
        sanctions_recall=1.0,
        latency_p50_ms=10.0,
        latency_p95_ms=20.0,
        total_cost_usd=5.0,   # non-zero cost
        eval_size=1,
        generated_at=datetime.now(tz=timezone.utc),
    )
    with pytest.raises(AssertionError, match="0.0"):
        save_metrics(bad_report, path=tmp_path / "m.json", checksums_path=tmp_path / "cs.sha256")


# ── Integration: frozen artifact ──────────────────────────────────────────────

@_skip_metrics
def test_metrics_schema_valid() -> None:
    """metrics_baseline.json parses as MetricsReport without ValidationError."""
    report = MetricsReport.model_validate_json(_METRICS_PATH.read_text(encoding="utf-8"))
    assert report.eval_size == 90
    assert report.total_cost_usd == 0.0
    assert report.generated_at.tzinfo is not None


@_skip_metrics
def test_metrics_checksum_recorded() -> None:
    """artifacts/checksums.sha256 contains an entry for metrics_baseline.json."""
    from aml_copilot.utils.checksum import _to_key
    checksum_file = _ROOT / "artifacts/checksums.sha256"
    assert checksum_file.exists()
    text = checksum_file.read_text(encoding="utf-8")
    # checksums.sha256 stores repo-relative keys, not absolute paths
    assert _to_key(_METRICS_PATH) in text, (
        f"No checksum entry found for {_METRICS_PATH}"
    )


@_skip_metrics
def test_metrics_values_in_range() -> None:
    """All metric values are in their expected [0, 1] or positive ranges."""
    report = MetricsReport.model_validate_json(_METRICS_PATH.read_text(encoding="utf-8"))
    assert 0.0 <= report.disposition_accuracy      <= 1.0
    assert 0.0 <= report.false_clear_rate_weighted <= 1.0
    assert 0.0 <= report.sanctions_precision       <= 1.0
    assert 0.0 <= report.sanctions_recall          <= 1.0
    assert report.latency_p50_ms > 0
    assert report.latency_p95_ms >= report.latency_p50_ms


@_skip_inputs
def test_single_command_exits_0(tmp_path) -> None:
    """python -m aml_copilot.step8_metrics.metrics exits 0 and writes metrics_baseline.json."""
    import subprocess

    out = tmp_path / "metrics_baseline.json"
    cs  = tmp_path / "checksums.sha256"
    proc = subprocess.run(
        [
            sys.executable, "-m", "aml_copilot.step8_metrics.metrics",
            "--eval",      str(_EVAL_PATH),
            "--results",   str(_RESULTS_PATH),
            "--out",       str(out),
            "--checksums", str(cs),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"metrics CLI exited {proc.returncode}\nstderr: {proc.stderr[-1000:]}"
    )
    assert out.exists(), "metrics_baseline.json was not written"
    report = MetricsReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert report.eval_size == 90
    assert report.total_cost_usd == 0.0
