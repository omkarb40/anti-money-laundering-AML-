"""
Tests for Phase 2.5 — LangGraph agent evaluation (mocked-LLM mode).

Unit tests use synthetic data and run with no raw data on disk.
Integration tests read the committed fixtures (eval.jsonl, results.jsonl,
metrics_baseline.json) and are skipped if those files are absent.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from aml_copilot.schemas import (
    AnomalyScore,
    CaseResult,
    EvalCase,
    MetricsReport,
    RuleFiring,
    SanctionsHit,
)
from aml_copilot.phase2_eval.run_langgraph_eval import (
    AGENT_ANOMALY_PCT_THRESHOLD,
    AGENT_MIN_RULE_SEV_FOR_OVERRIDE,
    HUMAN_REVIEW_ANOMALY_THRESHOLD,
    AMLAgentState,
    Phase2CaseResult,
    _load_baseline_results,
    _load_eval_cases,
    _mock_llm_call,
    build_graph,
    finalize_node,
    llm_decide_node,
    prepare_evidence_node,
    run,
)
from aml_copilot.phase2_eval.metrics import (
    Phase2MetricsReport,
    compute_phase2_metrics,
    save_phase2_metrics,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT           = Path(__file__).parents[1]
_EVAL_PATH      = _ROOT / "data/fixtures/eval.jsonl"
_BASELINE_PATH  = _ROOT / "artifacts/results.jsonl"
_METRICS_PATH   = _ROOT / "artifacts/metrics_baseline.json"
_PHASE2_OUT     = _ROOT / "artifacts/phase2_langgraph_results.jsonl"

_fixtures_present = pytest.mark.skipif(
    not (_EVAL_PATH.exists() and _BASELINE_PATH.exists()),
    reason="eval.jsonl or results.jsonl not on disk",
)
_baseline_metrics_present = pytest.mark.skipif(
    not _METRICS_PATH.exists(),
    reason="artifacts/metrics_baseline.json not on disk",
)


# ── Synthetic builders ────────────────────────────────────────────────────────

def _make_anomaly(percentile: float, is_flagged: bool = False) -> dict[str, Any]:
    return {
        "account_id": "TEST",
        "score": percentile * 5.0,
        "percentile": percentile,
        "is_flagged": is_flagged,
        "excluded_features": [],
    }


def _make_rule(rule_id: str, severity: int) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "severity": severity,
        "account_id": "TEST",
        "evidence": {},
        "window_start": "2022-09-01T00:00:00",
        "window_end": "2022-09-02T00:00:00",
    }


def _make_sanctions(score: float) -> dict[str, Any]:
    return {
        "account_id": "TEST",
        "assigned_name": "Test Name",
        "ofac_uid": "12345",
        "list_source": "SDN",
        "match_score": score,
        "scorer_used": "jaro_winkler",
        "matched_name_type": "canonical",
    }


def _make_evidence(
    sanctions_score: float = 0.0,
    rule_severity: int = 0,
    anomaly_pct: float = 0.5,
    is_flagged: bool = False,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "anomaly_score": _make_anomaly(anomaly_pct, is_flagged),
        "baseline_disposition": "CLEAR",
        "baseline_reason": "clear",
    }
    if sanctions_score > 0:
        evidence["sanctions_hits"] = [_make_sanctions(sanctions_score)]
    else:
        evidence["sanctions_hits"] = []
    if rule_severity > 0:
        evidence["rule_firings"] = [_make_rule("TEST_001", rule_severity)]
    else:
        evidence["rule_firings"] = []
    return evidence


def _eval_case(
    case_id: str,
    gold_label: str = "ESCALATE",
    case_type: str = "ibm_labeled",
    severity_band: int | None = None,
) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        account_id=f"ACC{case_id}",
        gold_label=gold_label,
        case_type=case_type,
        severity_band=severity_band,
        relevant_txn_ids=[],
        notes="unit test",
    )


def _phase2_result(
    case_id: str,
    disposition: str,
    baseline_disposition: str = "ESCALATE",
    agent_override: bool = False,
    human_review_flagged: bool = False,
    latency_ms: float = 1.0,
    tokens_used: int = 0,
    cost_usd: float = 0.0,
) -> Phase2CaseResult:
    return Phase2CaseResult(
        case_id=case_id,
        account_id=f"ACC{case_id}",
        disposition=disposition,
        decision_reason="test",
        sanctions_hits=[],
        rule_firings=[],
        anomaly_score=None,
        latency_ms=latency_ms,
        agent_reasoning="synthetic",
        agent_override=agent_override,
        baseline_disposition=baseline_disposition,
        human_review_flagged=human_review_flagged,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
    )


def _baseline_metrics(**overrides) -> MetricsReport:
    defaults = dict(
        disposition_accuracy=0.75,
        false_clear_rate_weighted=0.20,
        sanctions_precision=1.0,
        sanctions_recall=1.0,
        latency_p50_ms=50.0,
        latency_p95_ms=60.0,
        total_cost_usd=0.0,
        eval_size=90,
        generated_at=datetime.now(tz=timezone.utc),
    )
    defaults.update(overrides)
    return MetricsReport(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
# Mock LLM unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMockLLM:
    def test_branch1_sanctions_escalates(self):
        """Sanctions score >= 0.90 → ESCALATE, no human review."""
        ev = _make_evidence(sanctions_score=0.95)
        result = _mock_llm_call(ev)
        assert result["disposition"] == "ESCALATE"
        assert result["decision_reason"] == "sanctions_or_critical_rule"
        assert result["human_review"] is False

    def test_branch1_sanctions_exact_threshold_escalates(self):
        """Exactly 0.90 triggers Branch 1."""
        ev = _make_evidence(sanctions_score=0.90)
        assert _mock_llm_call(ev)["disposition"] == "ESCALATE"

    def test_branch1_sanctions_below_threshold_no_trigger(self):
        """Score 0.89 alone does not escalate from Branch 1."""
        ev = _make_evidence(sanctions_score=0.89, rule_severity=0, anomaly_pct=0.5)
        result = _mock_llm_call(ev)
        assert result["disposition"] == "CLEAR"

    def test_branch1_severity3_escalates(self):
        """Severity-3 rule → ESCALATE."""
        ev = _make_evidence(rule_severity=3, anomaly_pct=0.3)
        result = _mock_llm_call(ev)
        assert result["disposition"] == "ESCALATE"
        assert result["decision_reason"] == "sanctions_or_critical_rule"

    def test_branch2_agent_override_high_anomaly_sev2(self):
        """anomaly_pct >= 0.90 AND severity 2 → ESCALATE (agent extension)."""
        ev = _make_evidence(
            rule_severity=2,
            anomaly_pct=AGENT_ANOMALY_PCT_THRESHOLD,
            is_flagged=False,
        )
        result = _mock_llm_call(ev)
        assert result["disposition"] == "ESCALATE"
        assert result["decision_reason"] == "agent_anomaly_plus_elevated_rule"
        assert result["human_review"] is True

    def test_branch2_below_anomaly_threshold_no_override(self):
        """anomaly_pct = 0.89 with sev=2 does NOT hit Branch 2."""
        ev = _make_evidence(rule_severity=2, anomaly_pct=0.89, is_flagged=False)
        result = _mock_llm_call(ev)
        assert result["disposition"] == "CLEAR"

    def test_branch2_high_anomaly_no_rule_no_override(self):
        """anomaly_pct >= 0.90 but rule_severity 0 does NOT hit Branch 2."""
        ev = _make_evidence(rule_severity=0, anomaly_pct=0.95, is_flagged=False)
        result = _mock_llm_call(ev)
        assert result["disposition"] == "CLEAR"

    def test_branch3_clear_with_human_review_flag(self):
        """anomaly_pct > 0.85 with no escalation trigger → CLEAR but flagged."""
        ev = _make_evidence(
            rule_severity=1,
            anomaly_pct=HUMAN_REVIEW_ANOMALY_THRESHOLD + 0.01,
            is_flagged=False,
        )
        result = _mock_llm_call(ev)
        assert result["disposition"] == "CLEAR"
        assert result["human_review"] is True

    def test_branch3_low_risk_no_review_flag(self):
        """Low anomaly, no rules → CLEAR, no human review."""
        ev = _make_evidence(rule_severity=0, anomaly_pct=0.3)
        result = _mock_llm_call(ev)
        assert result["disposition"] == "CLEAR"
        assert result["human_review"] is False

    def test_mock_llm_deterministic(self):
        """Same input always produces identical output."""
        ev = _make_evidence(rule_severity=2, anomaly_pct=0.92)
        assert _mock_llm_call(ev) == _mock_llm_call(ev)

    def test_empty_evidence_no_crash(self):
        """Empty evidence dict → CLEAR without exception."""
        result = _mock_llm_call({})
        assert result["disposition"] == "CLEAR"


# ══════════════════════════════════════════════════════════════════════════════
# LangGraph node unit tests
# ══════════════════════════════════════════════════════════════════════════════

def _base_state(baseline_json: str | None = None) -> AMLAgentState:
    if baseline_json is None:
        dummy = {
            "case_id": "X", "account_id": "A", "disposition": "CLEAR",
            "decision_reason": "clear", "sanctions_hits": [], "rule_firings": [],
            "anomaly_score": None, "latency_ms": 10.0,
        }
        baseline_json = json.dumps(dummy)
    return AMLAgentState(
        case_id="X",
        account_id="A",
        case_type="ibm_labeled",
        notes="",
        baseline_result_json=baseline_json,
        evidence={},
        agent_disposition="",
        agent_decision_reason="",
        agent_reasoning="",
        agent_confidence=0.0,
        human_review_flagged=False,
        tokens_used=0,
        cost_usd=0.0,
    )


class TestLangGraphNodes:
    def test_prepare_evidence_populates_evidence(self):
        """prepare_evidence_node fills evidence dict from baseline JSON."""
        state = _base_state()
        update = prepare_evidence_node(state)
        assert "evidence" in update
        assert "baseline_disposition" in update["evidence"]
        assert "sanctions_hits" in update["evidence"]

    def test_llm_decide_sets_disposition(self):
        """llm_decide_node writes agent_disposition to state."""
        state = _base_state()
        # Inject prepared evidence manually
        state = AMLAgentState(**{**state, "evidence": _make_evidence()})
        update = llm_decide_node(state)
        assert update["agent_disposition"] in ("ESCALATE", "CLEAR")
        assert update["agent_reasoning"]
        assert update["tokens_used"] == 0
        assert update["cost_usd"] == 0.0

    def test_finalize_node_is_noop(self):
        """finalize_node returns an empty dict."""
        state = _base_state()
        update = finalize_node(state)
        assert update == {}

    def test_build_graph_compiles(self):
        """build_graph() returns a compiled graph without raising."""
        graph = build_graph()
        assert graph is not None

    def test_graph_invoke_single_case(self):
        """graph.invoke() on one case returns a final state with disposition."""
        graph = build_graph()
        state = _base_state()
        result = graph.invoke(state)
        assert result["agent_disposition"] in ("ESCALATE", "CLEAR")
        assert result["agent_reasoning"]

    def test_graph_sanctions_hit_escalates(self):
        """Graph: sanctions hit >= 0.90 in baseline JSON → ESCALATE."""
        baseline = {
            "case_id": "X", "account_id": "A", "disposition": "ESCALATE",
            "decision_reason": "sanctions_or_critical_rule",
            "sanctions_hits": [_make_sanctions(0.95)],
            "rule_firings": [],
            "anomaly_score": _make_anomaly(0.3),
            "latency_ms": 10.0,
        }
        graph = build_graph()
        state = _base_state(json.dumps(baseline))
        result = graph.invoke(state)
        assert result["agent_disposition"] == "ESCALATE"

    def test_graph_branch2_override(self):
        """Graph: anomaly 0.93 + sev-2 rule overrides a CLEAR baseline."""
        baseline = {
            "case_id": "X", "account_id": "A", "disposition": "CLEAR",
            "decision_reason": "clear",
            "sanctions_hits": [],
            "rule_firings": [_make_rule("FAN_OUT_001", 2)],
            "anomaly_score": _make_anomaly(0.93, is_flagged=False),
            "latency_ms": 10.0,
        }
        graph = build_graph()
        state = _base_state(json.dumps(baseline))
        result = graph.invoke(state)
        assert result["agent_disposition"] == "ESCALATE"
        assert result["agent_decision_reason"] == "agent_anomaly_plus_elevated_rule"
        assert result["human_review_flagged"] is True


# ══════════════════════════════════════════════════════════════════════════════
# Phase2CaseResult schema tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase2CaseResultSchema:
    def test_valid_result_parses(self):
        r = _phase2_result("T1", "ESCALATE")
        assert r.case_id == "T1"
        assert r.tokens_used == 0
        assert r.cost_usd == 0.0

    def test_round_trip_json(self):
        """Phase2CaseResult serialises and deserialises identically."""
        r = _phase2_result("T2", "CLEAR", agent_override=True)
        r2 = Phase2CaseResult.model_validate_json(r.model_dump_json())
        assert r == r2

    def test_agent_override_false_when_same_as_baseline(self):
        r = _phase2_result("T3", "ESCALATE", baseline_disposition="ESCALATE", agent_override=False)
        assert r.agent_override is False

    def test_agent_override_true_when_different(self):
        r = _phase2_result("T4", "ESCALATE", baseline_disposition="CLEAR", agent_override=True)
        assert r.agent_override is True


# ══════════════════════════════════════════════════════════════════════════════
# Phase2MetricsReport unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase2Metrics:
    def _make_pairs(self, n_correct: int, n_total: int):
        """n_correct correct escalations, rest are FNs."""
        evals = [_eval_case(f"E{i}", "ESCALATE", severity_band=1) for i in range(n_total)]
        results = [
            _phase2_result(
                f"E{i}",
                "ESCALATE" if i < n_correct else "CLEAR",
                baseline_disposition="ESCALATE" if i < n_correct else "CLEAR",
                agent_override=False,
            )
            for i in range(n_total)
        ]
        return evals, results

    def test_perfect_accuracy(self):
        evals = [_eval_case("A", "ESCALATE", severity_band=1), _eval_case("B", "CLEAR")]
        results = [
            _phase2_result("A", "ESCALATE", baseline_disposition="ESCALATE"),
            _phase2_result("B", "CLEAR",    baseline_disposition="CLEAR"),
        ]
        report = compute_phase2_metrics(results, evals, _baseline_metrics())
        assert report.disposition_accuracy == pytest.approx(1.0)

    def test_human_review_rate_correct(self):
        evals = [_eval_case(f"C{i}", "CLEAR") for i in range(4)]
        results = [
            _phase2_result(f"C{i}", "CLEAR", human_review_flagged=(i < 2))
            for i in range(4)
        ]
        report = compute_phase2_metrics(results, evals, _baseline_metrics())
        assert report.human_review_rate == pytest.approx(0.5)

    def test_override_rate_correct(self):
        evals = [_eval_case(f"X{i}", "CLEAR") for i in range(5)]
        results = [
            _phase2_result(f"X{i}", "CLEAR", agent_override=(i < 1))
            for i in range(5)
        ]
        report = compute_phase2_metrics(results, evals, _baseline_metrics())
        assert report.override_rate == pytest.approx(0.2)

    def test_delta_accuracy_positive_when_improved(self):
        evals = [_eval_case("A", "ESCALATE", severity_band=1), _eval_case("B", "CLEAR")]
        results = [
            _phase2_result("A", "ESCALATE"),
            _phase2_result("B", "CLEAR"),
        ]
        baseline = _baseline_metrics(disposition_accuracy=0.5)
        report = compute_phase2_metrics(results, evals, baseline)
        assert report.delta_accuracy > 0.0

    def test_delta_fcr_negative_when_improved(self):
        evals = [_eval_case("A", "ESCALATE", severity_band=1)]
        results = [_phase2_result("A", "ESCALATE")]  # no FN
        baseline = _baseline_metrics(false_clear_rate_weighted=0.5)
        report = compute_phase2_metrics(results, evals, baseline)
        assert report.delta_false_clear_rate < 0.0

    def test_total_tokens_sum(self):
        evals = [_eval_case(f"T{i}", "CLEAR") for i in range(3)]
        results = [_phase2_result(f"T{i}", "CLEAR", tokens_used=10) for i in range(3)]
        report = compute_phase2_metrics(results, evals, _baseline_metrics())
        assert report.total_tokens_used == 30

    def test_total_cost_sum(self):
        evals = [_eval_case(f"T{i}", "CLEAR") for i in range(3)]
        results = [_phase2_result(f"T{i}", "CLEAR", cost_usd=0.005) for i in range(3)]
        report = compute_phase2_metrics(results, evals, _baseline_metrics())
        assert report.total_cost_usd == pytest.approx(0.015)

    def test_eval_size_matches(self):
        n = 6
        evals = [_eval_case(f"Z{i}", "CLEAR") for i in range(n)]
        results = [_phase2_result(f"Z{i}", "CLEAR") for i in range(n)]
        report = compute_phase2_metrics(results, evals, _baseline_metrics())
        assert report.eval_size == n

    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_phase2_metrics([], [_eval_case("A", "CLEAR")], _baseline_metrics())

    def test_empty_eval_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_phase2_metrics([_phase2_result("A", "CLEAR")], [], _baseline_metrics())

    def test_generated_at_is_utc(self):
        evals = [_eval_case("A", "CLEAR")]
        results = [_phase2_result("A", "CLEAR")]
        report = compute_phase2_metrics(results, evals, _baseline_metrics())
        assert report.generated_at.tzinfo is not None
        assert report.generated_at.utcoffset().total_seconds() == 0

    def test_save_loads_round_trip(self, tmp_path):
        evals = [_eval_case("A", "ESCALATE", severity_band=1)]
        results = [_phase2_result("A", "ESCALATE")]
        report = compute_phase2_metrics(results, evals, _baseline_metrics())
        out = tmp_path / "phase2_metrics.json"
        save_phase2_metrics(report, path=out)
        loaded = Phase2MetricsReport.model_validate_json(out.read_text())
        assert loaded.eval_size == report.eval_size
        assert loaded.override_rate == pytest.approx(report.override_rate)

    def test_mock_cost_is_zero_in_unit(self):
        evals = [_eval_case("A", "CLEAR")]
        results = [_phase2_result("A", "CLEAR")]
        report = compute_phase2_metrics(results, evals, _baseline_metrics())
        assert report.total_cost_usd == 0.0
        assert report.total_tokens_used == 0


# ══════════════════════════════════════════════════════════════════════════════
# Integration tests — require committed fixture files
# ══════════════════════════════════════════════════════════════════════════════

@_fixtures_present
class TestIntegration:
    def test_all_90_cases_run(self, tmp_path):
        """run() processes all 90 eval cases and returns Phase2CaseResult list."""
        out = tmp_path / "phase2_results.jsonl"
        results = run(
            eval_path=_EVAL_PATH,
            baseline_path=_BASELINE_PATH,
            out_path=out,
        )
        assert len(results) == 90

    def test_output_file_written(self, tmp_path):
        """run() writes phase2_langgraph_results.jsonl atomically."""
        out = tmp_path / "phase2_results.jsonl"
        run(
            eval_path=_EVAL_PATH,
            baseline_path=_BASELINE_PATH,
            out_path=out,
        )
        assert out.exists()
        rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert len(rows) == 90

    def test_all_results_schema_valid(self, tmp_path):
        """All 90 output rows parse as Phase2CaseResult without error."""
        out = tmp_path / "r.jsonl"
        results = run(_EVAL_PATH, _BASELINE_PATH, out)
        for r in results:
            assert isinstance(r, Phase2CaseResult)
            assert r.disposition in ("ESCALATE", "CLEAR")
            assert r.baseline_disposition in ("ESCALATE", "CLEAR")

    def test_override_rate_nonzero(self, tmp_path):
        """Agent overrides at least some baseline decisions."""
        out = tmp_path / "r.jsonl"
        results = run(_EVAL_PATH, _BASELINE_PATH, out)
        overrides = sum(1 for r in results if r.agent_override)
        # With anomaly-pct-90 + sev-2 override logic we expect a few overrides
        assert overrides >= 1, "Expected at least one baseline override"

    def test_tokens_and_cost_zero_in_mock(self, tmp_path):
        """Mock mode: all cases report 0 tokens and $0.00 cost."""
        out = tmp_path / "r.jsonl"
        results = run(_EVAL_PATH, _BASELINE_PATH, out)
        assert all(r.tokens_used == 0 for r in results)
        assert all(r.cost_usd == 0.0 for r in results)

    def test_unique_case_ids(self, tmp_path):
        """No duplicate case_ids in the output."""
        out = tmp_path / "r.jsonl"
        results = run(_EVAL_PATH, _BASELINE_PATH, out)
        ids = [r.case_id for r in results]
        assert len(ids) == len(set(ids)), "Duplicate case_ids in phase2 results"

    def test_latency_ms_positive(self, tmp_path):
        """All latency values are positive."""
        out = tmp_path / "r.jsonl"
        results = run(_EVAL_PATH, _BASELINE_PATH, out)
        assert all(r.latency_ms > 0 for r in results)

    @_baseline_metrics_present
    def test_phase2_accuracy_improves_baseline(self, tmp_path):
        """Phase 2.5 agent accuracy >= deterministic baseline accuracy."""
        from aml_copilot.schemas import MetricsReport as BaseMetrics
        out = tmp_path / "r.jsonl"
        results = run(_EVAL_PATH, _BASELINE_PATH, out)
        eval_cases = _load_eval_cases(_EVAL_PATH)
        baseline_report = BaseMetrics.model_validate_json(
            _METRICS_PATH.read_text(encoding="utf-8")
        )
        from aml_copilot.phase2_eval.metrics import compute_phase2_metrics
        report = compute_phase2_metrics(results, eval_cases, baseline_report)
        # Mock LLM's Branch 2 should fix some FNs → delta_accuracy ≥ 0
        assert report.delta_accuracy >= 0.0, (
            f"Phase2 accuracy {report.disposition_accuracy:.4f} is lower than "
            f"baseline {baseline_report.disposition_accuracy:.4f}"
        )

    @_baseline_metrics_present
    def test_phase2_fcr_not_worse_than_baseline(self, tmp_path):
        """Phase 2.5 weighted false-clear rate ≤ baseline (or at worst equal)."""
        from aml_copilot.schemas import MetricsReport as BaseMetrics
        out = tmp_path / "r.jsonl"
        results = run(_EVAL_PATH, _BASELINE_PATH, out)
        eval_cases = _load_eval_cases(_EVAL_PATH)
        baseline_report = BaseMetrics.model_validate_json(
            _METRICS_PATH.read_text(encoding="utf-8")
        )
        from aml_copilot.phase2_eval.metrics import compute_phase2_metrics
        report = compute_phase2_metrics(results, eval_cases, baseline_report)
        assert report.delta_false_clear_rate <= 0.0, (
            f"Phase2 FCR {report.false_clear_rate_weighted:.4f} is worse than "
            f"baseline {baseline_report.false_clear_rate_weighted:.4f}"
        )

    def test_human_review_rate_in_range(self, tmp_path):
        """human_review_rate is in [0, 1]."""
        out = tmp_path / "r.jsonl"
        results = run(_EVAL_PATH, _BASELINE_PATH, out)
        eval_cases = _load_eval_cases(_EVAL_PATH)
        from aml_copilot.phase2_eval.metrics import compute_phase2_metrics
        report = compute_phase2_metrics(results, eval_cases, _baseline_metrics())
        assert 0.0 <= report.human_review_rate <= 1.0

    def test_cli_exits_0(self, tmp_path):
        """python -m aml_copilot.phase2_eval.run_langgraph_eval exits 0."""
        import subprocess
        out = tmp_path / "phase2_results.jsonl"
        proc = subprocess.run(
            [
                sys.executable,
                "-m", "aml_copilot.phase2_eval.run_langgraph_eval",
                "--eval",     str(_EVAL_PATH),
                "--baseline", str(_BASELINE_PATH),
                "--out",      str(out),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"CLI exited {proc.returncode}\nstderr:\n{proc.stderr[-1500:]}"
        )
        assert out.exists()
