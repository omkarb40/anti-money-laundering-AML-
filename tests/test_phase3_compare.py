"""
Phase 3 M1–M3 tests — shared mock LLM, AMLAgentRunner protocol, Phase 3 schemas,
LangGraph adapter, and CrewAI adapter.

M1 tests (1–28) run without raw HI-Small data.
M2 tests (29–35) require data/fixtures/eval.jsonl and artifacts/results.jsonl.
M3 tests (36–53) require data/fixtures/eval.jsonl and artifacts/results.jsonl.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

# ── Helpers shared across test classes ───────────────────────────────────────

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
    evidence["sanctions_hits"] = (
        [_make_sanctions(sanctions_score)] if sanctions_score > 0 else []
    )
    evidence["rule_firings"] = (
        [_make_rule("TEST_001", rule_severity)] if rule_severity > 0 else []
    )
    return evidence


# ── Frozen policy snapshots ───────────────────────────────────────────────────
# Plain literals derived from reading mock_llm.py — NOT derived by calling the
# function.  Any drift in the mock policy (threshold, confidence, reasoning
# string) will break the corresponding TestMockLLMPolicySnapshots test.

_MOCK_OUTPUT_KEYS: frozenset[str] = frozenset(
    {"disposition", "decision_reason", "reasoning", "confidence", "human_review"}
)

# Branch 1 — sanctions score 0.950 >= 0.90
_SNAP_A_SANCTIONS: dict[str, Any] = {
    "disposition": "ESCALATE",
    "decision_reason": "sanctions_or_critical_rule",
    "reasoning": (
        "Sanctions hit score 0.950 ≥ 0.90 threshold. "
        "OFAC compliance obligation requires immediate escalation."
    ),
    "confidence": 0.99,
    "human_review": False,
}

# Branch 2 — severity-3 rule, rule_id="TEST_001"
_SNAP_B_SEV3: dict[str, Any] = {
    "disposition": "ESCALATE",
    "decision_reason": "sanctions_or_critical_rule",
    "reasoning": (
        "Critical severity-3 rule(s) fired: ['TEST_001']. "
        "High-risk pattern warrants escalation."
    ),
    "confidence": 0.95,
    "human_review": False,
}

# Branch 3 — anomaly_pct=0.93 (score=4.65) + severity-2 rule
#   0.93 as 1% = "93.0%";  (1−0.93)×100 = 7.0
_SNAP_C_ANOMALY_RULE: dict[str, Any] = {
    "disposition": "ESCALATE",
    "decision_reason": "agent_anomaly_plus_elevated_rule",
    "reasoning": (
        "Anomaly at 93.0% percentile "
        "(robust-z 4.65) — top "
        "7.0% of account population — "
        "combined with severity-2 rule. "
        "Deterministic baseline requires the 99.5th-percentile flag; "
        "agent escalates at the 90th-percentile when paired with "
        "elevated rule evidence."
    ),
    "confidence": 0.78,
    "human_review": True,
}

# Branch 4 (elevated) — anomaly_pct=0.87 > 0.85, severity-1 rule (< 2, no Branch 3 trigger)
_SNAP_D_ELEVATED_CLEAR: dict[str, Any] = {
    "disposition": "CLEAR",
    "decision_reason": "clear",
    "reasoning": (
        "Anomaly at 87.0% percentile with max rule severity "
        "1. Below escalation threshold but elevated; "
        "flagging for human review."
    ),
    "confidence": 0.88,
    "human_review": True,
}

# Branch 4 (clean) — anomaly_pct=0.5 (default), no rules, no sanctions
_SNAP_E_CLEAN: dict[str, Any] = {
    "disposition": "CLEAR",
    "decision_reason": "clear",
    "reasoning": (
        "No significant risk indicators: anomaly 50.0% "
        "percentile, max rule severity 0. Case clears."
    ),
    "confidence": 0.88,
    "human_review": False,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Mock LLM — direct tests of mock_llm_call from phase3_compare.mock_llm
# ══════════════════════════════════════════════════════════════════════════════

class TestMockLLMCall:
    """Tests 1–10: verify mock_llm_call output for all branches."""

    def test_sanctions_hit_branch_escalates(self):
        """Branch 1: sanctions score >= 0.90 → ESCALATE, no human review."""
        from aml_copilot.phase3_compare.mock_llm import mock_llm_call
        ev = _make_evidence(sanctions_score=0.95)
        result = mock_llm_call(ev)
        assert result["disposition"] == "ESCALATE"
        assert result["decision_reason"] == "sanctions_or_critical_rule"
        assert result["human_review"] is False
        assert result["confidence"] == pytest.approx(0.99)

    def test_sanctions_exact_threshold_escalates(self):
        """Branch 1: score exactly 0.90 triggers escalation (inclusive >=)."""
        from aml_copilot.phase3_compare.mock_llm import mock_llm_call
        ev = _make_evidence(sanctions_score=0.90)
        assert mock_llm_call(ev)["disposition"] == "ESCALATE"

    def test_severity3_rule_branch_escalates(self):
        """Branch 2: severity-3 rule → ESCALATE, no human review."""
        from aml_copilot.phase3_compare.mock_llm import mock_llm_call
        ev = _make_evidence(rule_severity=3, anomaly_pct=0.3)
        result = mock_llm_call(ev)
        assert result["disposition"] == "ESCALATE"
        assert result["decision_reason"] == "sanctions_or_critical_rule"
        assert result["human_review"] is False
        assert result["confidence"] == pytest.approx(0.95)

    def test_anomaly_plus_elevated_rule_escalates(self):
        """Branch 3: anomaly_pct >= 0.90 AND severity >= 2 → ESCALATE, human_review True."""
        from aml_copilot.phase3_compare.mock_llm import mock_llm_call
        ev = _make_evidence(rule_severity=2, anomaly_pct=0.93)
        result = mock_llm_call(ev)
        assert result["disposition"] == "ESCALATE"
        assert result["decision_reason"] == "agent_anomaly_plus_elevated_rule"
        assert result["human_review"] is True
        assert result["confidence"] == pytest.approx(0.78)

    def test_anomaly_at_threshold_plus_elevated_rule_escalates(self):
        """Branch 3: anomaly_pct exactly 0.90 + severity 2 triggers (inclusive >=)."""
        from aml_copilot.phase3_compare.mock_llm import mock_llm_call, AGENT_ANOMALY_PCT_THRESHOLD
        ev = _make_evidence(rule_severity=2, anomaly_pct=AGENT_ANOMALY_PCT_THRESHOLD)
        assert mock_llm_call(ev)["disposition"] == "ESCALATE"

    def test_elevated_anomaly_without_sufficient_rule_clears(self):
        """Branch 4: anomaly_pct >= 0.90 but severity < 2 → CLEAR (branch 3 condition fails)."""
        from aml_copilot.phase3_compare.mock_llm import mock_llm_call
        ev = _make_evidence(rule_severity=1, anomaly_pct=0.93)
        result = mock_llm_call(ev)
        assert result["disposition"] == "CLEAR"
        assert result["human_review"] is True   # 0.93 > 0.85

    def test_elevated_anomaly_no_rule_human_review_flagged(self):
        """Branch 4: anomaly_pct > 0.85 with no rule → CLEAR, human_review True."""
        from aml_copilot.phase3_compare.mock_llm import mock_llm_call
        ev = _make_evidence(rule_severity=0, anomaly_pct=0.87)
        result = mock_llm_call(ev)
        assert result["disposition"] == "CLEAR"
        assert result["human_review"] is True

    def test_human_review_exact_boundary_not_flagged(self):
        """Branch 4: anomaly_pct == 0.85 is NOT flagged (strict >)."""
        from aml_copilot.phase3_compare.mock_llm import mock_llm_call, HUMAN_REVIEW_ANOMALY_THRESHOLD
        ev = _make_evidence(anomaly_pct=HUMAN_REVIEW_ANOMALY_THRESHOLD)
        result = mock_llm_call(ev)
        assert result["disposition"] == "CLEAR"
        assert result["human_review"] is False

    def test_clean_case_clears_no_human_review(self):
        """Branch 4 (clean): no signals → CLEAR, human_review False."""
        from aml_copilot.phase3_compare.mock_llm import mock_llm_call
        ev = _make_evidence(sanctions_score=0.0, rule_severity=0, anomaly_pct=0.3)
        result = mock_llm_call(ev)
        assert result["disposition"] == "CLEAR"
        assert result["human_review"] is False
        assert result["confidence"] == pytest.approx(0.88)

    def test_empty_evidence_dict_clears_without_error(self):
        """Empty dict → Branch 4 (clean), no exception."""
        from aml_copilot.phase3_compare.mock_llm import mock_llm_call
        result = mock_llm_call({})
        assert result["disposition"] == "CLEAR"
        assert result["human_review"] is False


# ══════════════════════════════════════════════════════════════════════════════
# 2. Policy snapshots — mock_llm_call and _mock_llm_call verified independently
#    against literal expected outputs for every decision branch
# ══════════════════════════════════════════════════════════════════════════════

class TestMockLLMPolicySnapshots:
    """
    Tests 11–15: frozen expected-output snapshots for every policy branch.

    Both public entry points (mock_llm_call from phase3_compare.mock_llm and
    the backward-compatible _mock_llm_call alias from phase2_eval.run_langgraph_eval)
    are verified independently against the same literal expected dict defined at
    module level above.  Any drift in either function body will break the
    corresponding test.

    The exact output-key contract (exactly five keys) is also checked for each
    scenario and each entry point.
    """

    def _assert_snapshot(
        self,
        evidence: dict[str, Any],
        expected: dict[str, Any],
    ) -> None:
        from aml_copilot.phase3_compare.mock_llm import mock_llm_call
        from aml_copilot.phase2_eval.run_langgraph_eval import _mock_llm_call

        out_p3 = mock_llm_call(evidence)
        out_p2 = _mock_llm_call(evidence)

        # Full-dict equality against the frozen literal snapshot
        assert out_p3 == expected, (
            f"mock_llm_call mismatch\n  got:      {out_p3}\n  expected: {expected}"
        )
        assert out_p2 == expected, (
            f"_mock_llm_call mismatch\n  got:      {out_p2}\n  expected: {expected}"
        )

        # Exact key-set contract: exactly the five documented keys, no more, no fewer
        assert set(out_p3.keys()) == _MOCK_OUTPUT_KEYS, (
            f"mock_llm_call key set: {set(out_p3.keys())}"
        )
        assert set(out_p2.keys()) == _MOCK_OUTPUT_KEYS, (
            f"_mock_llm_call key set: {set(out_p2.keys())}"
        )

    def test_snapshot_branch1_sanctions(self):
        """Branch 1: sanctions score 0.950 >= 0.90 — ESCALATE, confidence 0.99."""
        self._assert_snapshot(_make_evidence(sanctions_score=0.95), _SNAP_A_SANCTIONS)

    def test_snapshot_branch2_severity3(self):
        """Branch 2: severity-3 rule — ESCALATE, confidence 0.95, no human review."""
        self._assert_snapshot(_make_evidence(rule_severity=3, anomaly_pct=0.3), _SNAP_B_SEV3)

    def test_snapshot_branch3_anomaly_rule(self):
        """Branch 3: anomaly 0.93 + severity-2 rule — ESCALATE, human review True."""
        self._assert_snapshot(_make_evidence(rule_severity=2, anomaly_pct=0.93), _SNAP_C_ANOMALY_RULE)

    def test_snapshot_branch4_elevated_clear(self):
        """Branch 4 (elevated): anomaly 0.87 > 0.85, sev-1 rule — CLEAR, human review True."""
        self._assert_snapshot(_make_evidence(rule_severity=1, anomaly_pct=0.87), _SNAP_D_ELEVATED_CLEAR)

    def test_snapshot_branch4_clean_clear(self):
        """Branch 4 (clean): no material signals — CLEAR, human review False."""
        self._assert_snapshot(_make_evidence(), _SNAP_E_CLEAN)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Backward compatibility — all Phase 2 imports remain intact
# ══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Tests 16–18: names that test_phase2_eval.py imports must still work."""

    def test_mock_llm_alias_importable_and_callable(self):
        """_mock_llm_call is importable from run_langgraph_eval and callable."""
        from aml_copilot.phase2_eval.run_langgraph_eval import _mock_llm_call
        result = _mock_llm_call(_make_evidence())
        assert result["disposition"] in ("ESCALATE", "CLEAR")

    def test_all_threshold_constants_importable_from_phase2(self):
        """All five threshold constants (old + new aliases) importable from phase2."""
        from aml_copilot.phase2_eval.run_langgraph_eval import (  # noqa: F401
            AGENT_ANOMALY_PCT_THRESHOLD,
            AGENT_MIN_RULE_SEV_FOR_OVERRIDE,
            HUMAN_REVIEW_ANOMALY_THRESHOLD,
            HUMAN_REVIEW_ANOMALY_MIN,
            HUMAN_REVIEW_ANOMALY_MAX,
        )

    def test_threshold_constant_values_correct(self):
        """Constant values match the original Phase 2 thresholds."""
        from aml_copilot.phase2_eval.run_langgraph_eval import (
            AGENT_ANOMALY_PCT_THRESHOLD,
            AGENT_MIN_RULE_SEV_FOR_OVERRIDE,
            HUMAN_REVIEW_ANOMALY_THRESHOLD,
            HUMAN_REVIEW_ANOMALY_MIN,
            HUMAN_REVIEW_ANOMALY_MAX,
        )
        assert AGENT_ANOMALY_PCT_THRESHOLD == pytest.approx(0.90)
        assert AGENT_MIN_RULE_SEV_FOR_OVERRIDE == 2
        assert HUMAN_REVIEW_ANOMALY_THRESHOLD == pytest.approx(0.85)
        assert HUMAN_REVIEW_ANOMALY_MIN == pytest.approx(0.85)
        assert HUMAN_REVIEW_ANOMALY_MAX == pytest.approx(0.90)


# ══════════════════════════════════════════════════════════════════════════════
# 4. AMLAgentRunner Protocol
# ══════════════════════════════════════════════════════════════════════════════

class TestAMLAgentRunnerProtocol:
    """Tests 19–23: Protocol structure and runtime_checkable behaviour."""

    def test_protocol_importable(self):
        from aml_copilot.phase3_compare.protocol import AMLAgentRunner  # noqa: F401

    def test_protocol_version_is_1_0(self):
        from aml_copilot.phase3_compare.protocol import PROTOCOL_VERSION
        assert PROTOCOL_VERSION == "1.0"

    def test_conforming_class_passes_isinstance(self):
        """A class with framework_name and run() satisfies the Protocol at runtime."""
        from aml_copilot.phase3_compare.protocol import AMLAgentRunner
        from aml_copilot.schemas import Phase3CaseResult

        class _GoodRunner:
            framework_name: str = "test_framework"

            def run(self, eval_path: Path, baseline_path: Path) -> list[Phase3CaseResult]:
                return []

        assert isinstance(_GoodRunner(), AMLAgentRunner)

    def test_missing_run_method_fails_isinstance(self):
        """An object without run() does not satisfy the Protocol."""
        from aml_copilot.phase3_compare.protocol import AMLAgentRunner

        class _NoRun:
            framework_name: str = "test_framework"

        assert not isinstance(_NoRun(), AMLAgentRunner)

    def test_missing_framework_name_fails_isinstance(self):
        """An object without framework_name does not satisfy the Protocol."""
        from aml_copilot.phase3_compare.protocol import AMLAgentRunner
        from aml_copilot.schemas import Phase3CaseResult

        class _NoName:
            def run(self, eval_path: Path, baseline_path: Path) -> list[Phase3CaseResult]:
                return []

        assert not isinstance(_NoName(), AMLAgentRunner)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Phase 3 schemas
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase3Schemas:
    """Tests 24–27: Pydantic model construction, JSON round-trips, field constraints."""

    def _minimal_phase3_result(self, framework: str = "langgraph") -> dict[str, Any]:
        return {
            "case_id": "CASE_001",
            "account_id": "ACC_001",
            "framework": framework,
            "disposition": "ESCALATE",
            "decision_reason": "sanctions_or_critical_rule",
            "sanctions_hits": [],
            "rule_firings": [],
            "anomaly_score": None,
            "latency_ms": 1.5,
            "agent_reasoning": "Test reasoning.",
            "agent_override": False,
            "baseline_disposition": "ESCALATE",
            "human_review_flagged": False,
        }

    def test_phase3_case_result_roundtrip(self):
        """Phase3CaseResult serialises and deserialises identically."""
        from aml_copilot.schemas import Phase3CaseResult
        r = Phase3CaseResult(**self._minimal_phase3_result())
        r2 = Phase3CaseResult.model_validate_json(r.model_dump_json())
        assert r == r2
        assert r2.framework == "langgraph"
        assert r2.tokens_used == 0
        assert r2.cost_usd == 0.0

    def test_phase3_framework_metrics_roundtrip(self):
        """Phase3FrameworkMetrics serialises and deserialises identically."""
        from aml_copilot.schemas import Phase3FrameworkMetrics
        m = Phase3FrameworkMetrics(
            framework="langgraph",
            disposition_accuracy=0.7889,
            false_clear_rate_weighted=0.1722,
            override_rate=0.0556,
            human_review_rate=0.1667,
            latency_p50_ms=0.47,
            latency_p95_ms=0.61,
            loc=250,
            total_cost_usd=0.0,
            eval_size=90,
        )
        m2 = Phase3FrameworkMetrics.model_validate_json(m.model_dump_json())
        assert m == m2
        assert m2.framework == "langgraph"

    def test_phase3_comparison_metrics_roundtrip(self):
        """Phase3ComparisonMetrics serialises and deserialises identically."""
        from aml_copilot.schemas import Phase3ComparisonMetrics, Phase3FrameworkMetrics
        fw = Phase3FrameworkMetrics(
            framework="langgraph",
            disposition_accuracy=0.7889,
            false_clear_rate_weighted=0.1722,
            override_rate=0.0556,
            human_review_rate=0.1667,
            latency_p50_ms=0.47,
            latency_p95_ms=0.61,
            loc=250,
            total_cost_usd=0.0,
            eval_size=90,
        )
        cm = Phase3ComparisonMetrics(
            generated_at=datetime.now(tz=timezone.utc),
            eval_size=90,
            protocol_version="1.0",
            phase1_accuracy=0.7556,
            phase2_accuracy=0.7889,
            frameworks=[fw],
            all_dispositions_agree=True,
            all_reasoning_agree=True,
            all_human_review_flags_agree=True,
            all_costs_zero=True,
            all_tokens_zero=True,
            comparison_passed=True,
        )
        cm2 = Phase3ComparisonMetrics.model_validate_json(cm.model_dump_json())
        assert cm2.protocol_version == "1.0"
        assert cm2.all_dispositions_agree is True
        assert cm2.all_reasoning_agree is True
        assert cm2.all_human_review_flags_agree is True
        assert cm2.comparison_passed is True
        assert cm2.generated_at.tzinfo is not None
        assert len(cm2.frameworks) == 1
        assert cm2.frameworks[0].framework == "langgraph"

    def test_phase3_case_result_requires_framework(self):
        """Omitting the required 'framework' field raises pydantic.ValidationError."""
        from pydantic import ValidationError
        from aml_copilot.schemas import Phase3CaseResult

        payload = {k: v for k, v in self._minimal_phase3_result().items() if k != "framework"}
        with pytest.raises(ValidationError):
            Phase3CaseResult(**payload)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Circular-import smoke test
# ══════════════════════════════════════════════════════════════════════════════

class TestCircularImport:
    """Test 28: all Phase 2/3 modules import cleanly with no circular-import errors."""

    def test_no_circular_import(self):
        """Importing all Phase 2/3 modules in sequence raises no ImportError."""
        import aml_copilot.phase3_compare  # noqa: F401
        import aml_copilot.phase3_compare.mock_llm  # noqa: F401
        import aml_copilot.phase3_compare.protocol  # noqa: F401
        import aml_copilot.schemas  # noqa: F401
        import aml_copilot.phase2_eval.run_langgraph_eval  # noqa: F401

        # Backward-compatible alias and all constants remain accessible post-import
        from aml_copilot.phase2_eval.run_langgraph_eval import (
            _mock_llm_call,
            AGENT_ANOMALY_PCT_THRESHOLD,
            AGENT_MIN_RULE_SEV_FOR_OVERRIDE,
            HUMAN_REVIEW_ANOMALY_THRESHOLD,
            HUMAN_REVIEW_ANOMALY_MIN,
            HUMAN_REVIEW_ANOMALY_MAX,
        )
        assert callable(_mock_llm_call)
        assert AGENT_ANOMALY_PCT_THRESHOLD == pytest.approx(0.90)
        assert AGENT_MIN_RULE_SEV_FOR_OVERRIDE == 2
        assert HUMAN_REVIEW_ANOMALY_THRESHOLD == pytest.approx(0.85)
        assert HUMAN_REVIEW_ANOMALY_MIN == pytest.approx(0.85)
        assert HUMAN_REVIEW_ANOMALY_MAX == pytest.approx(0.90)


# ══════════════════════════════════════════════════════════════════════════════
# M2 — LangGraph adapter tests
# Require: data/fixtures/eval.jsonl  artifacts/results.jsonl
# ══════════════════════════════════════════════════════════════════════════════

_ROOT = Path(__file__).parents[1]
_EVAL_PATH = _ROOT / "data/fixtures/eval.jsonl"
_BASELINE_PATH = _ROOT / "artifacts/results.jsonl"
_P2_RESULTS_PATH = _ROOT / "artifacts/phase2_langgraph_results.jsonl"
_EXPECTED_P2_ACCURACY: float = 0.7888888888888889
_ACCURACY_TOL: float = 1e-6


@pytest.fixture(scope="session")
def _fixture_files_available() -> bool:
    """Return True if both fixture files required for integration tests exist on disk."""
    return _EVAL_PATH.exists() and _BASELINE_PATH.exists()


@pytest.fixture
def _require_fixture_files(_fixture_files_available: bool) -> None:
    """Skip the calling test when fixture files are not present."""
    if not _fixture_files_available:
        pytest.skip(
            "Fixture files not found: data/fixtures/eval.jsonl and/or "
            "artifacts/results.jsonl — run the baseline pipeline first."
        )


@pytest.fixture(scope="module")
def _langgraph_results(_fixture_files_available: bool) -> list:
    """Run the LangGraph adapter once; share results across all M2 tests."""
    if not _fixture_files_available:
        pytest.skip("Fixture files not available")
    from aml_copilot.phase3_compare.langgraph_runner import LangGraphRunner
    return LangGraphRunner().run(_EVAL_PATH, _BASELINE_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# 7. LangGraph adapter — protocol, schema, parity, accuracy
# ══════════════════════════════════════════════════════════════════════════════

class TestLangGraphRunner:
    """Tests 29–35: Phase 3 M2 LangGraph adapter."""
    pytestmark = [pytest.mark.integration, pytest.mark.compare, pytest.mark.slow]

    def test_langgraph_runner_protocol(self):
        """LangGraphRunner satisfies AMLAgentRunner at runtime."""
        from aml_copilot.phase3_compare.protocol import AMLAgentRunner
        from aml_copilot.phase3_compare.langgraph_runner import LangGraphRunner
        assert isinstance(LangGraphRunner(), AMLAgentRunner)

    def test_langgraph_runner_returns_90(self, _langgraph_results):
        """Adapter returns exactly 90 results."""
        assert len(_langgraph_results) == 90

    def test_langgraph_runner_framework_tag(self, _langgraph_results):
        """Every result carries framework == 'langgraph'."""
        assert all(r.framework == "langgraph" for r in _langgraph_results)

    def test_langgraph_runner_schema(self, _langgraph_results):
        """Every returned object is a valid Phase3CaseResult."""
        from aml_copilot.schemas import Phase3CaseResult
        for r in _langgraph_results:
            assert isinstance(r, Phase3CaseResult)

    def test_langgraph_runner_unique_case_ids(self, _langgraph_results):
        """No duplicate case IDs in the output."""
        ids = [r.case_id for r in _langgraph_results]
        assert len(ids) == len(set(ids))

    def test_langgraph_runner_matches_phase2(self, _langgraph_results):
        """Adapter output matches existing Phase 2 results for all required fields."""
        if not _P2_RESULTS_PATH.exists():
            pytest.skip(f"Phase 2 results not found: {_P2_RESULTS_PATH}")

        from aml_copilot.phase2_eval.run_langgraph_eval import Phase2CaseResult
        p2_map: dict[str, Phase2CaseResult] = {}
        with open(_P2_RESULTS_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = Phase2CaseResult.model_validate_json(line)
                    p2_map[r.case_id] = r

        scalar_fields = [
            "disposition",
            "decision_reason",
            "human_review_flagged",
            "agent_override",
            "cost_usd",
            "tokens_used",
        ]
        nested_fields = ["sanctions_hits", "rule_firings", "anomaly_score"]

        for p3 in _langgraph_results:
            assert p3.case_id in p2_map, f"case_id {p3.case_id!r} absent from Phase 2 results"
            p2 = p2_map[p3.case_id]

            for field in scalar_fields:
                p3_val = getattr(p3, field)
                p2_val = getattr(p2, field)
                assert p3_val == p2_val, (
                    f"{p3.case_id}: {field} mismatch — "
                    f"adapter={p3_val!r}  phase2={p2_val!r}"
                )

            for field in nested_fields:
                p3_val = getattr(p3, field)
                p2_val = getattr(p2, field)
                assert p3_val == p2_val, (
                    f"{p3.case_id}: {field} mismatch — "
                    f"adapter={p3_val!r}  phase2={p2_val!r}"
                )

    def test_langgraph_runner_accuracy(self, _langgraph_results):
        """Disposition accuracy matches the committed Phase 2 metric within tolerance."""
        from aml_copilot.schemas import EvalCase
        gold: dict[str, str] = {}
        with open(_EVAL_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    c = EvalCase.model_validate_json(line)
                    gold[c.case_id] = c.gold_label

        correct = sum(1 for r in _langgraph_results if gold.get(r.case_id) == r.disposition)
        accuracy = correct / len(_langgraph_results)
        assert accuracy == pytest.approx(_EXPECTED_P2_ACCURACY, abs=_ACCURACY_TOL), (
            f"Accuracy {accuracy:.6f} differs from expected {_EXPECTED_P2_ACCURACY:.6f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 8. CrewAI adapter — unit, integration, parity (M3, tests 36–53)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def _crewai_results(_fixture_files_available: bool) -> list:
    """Run the CrewAI adapter once; share results across all M3 tests."""
    if not _fixture_files_available:
        pytest.skip("Fixture files not available")
    from aml_copilot.phase3_compare.crewai_runner import CrewAIRunner
    return CrewAIRunner(verbose=False).run(_EVAL_PATH, _BASELINE_PATH)


class TestCrewAIRunner:
    """Tests 36–53: Phase 3 M3 CrewAI adapter."""
    pytestmark = [pytest.mark.integration, pytest.mark.compare, pytest.mark.slow]

    # ── Protocol and class-level contracts ────────────────────────────────────

    def test_crewai_runner_implements_protocol(self):
        """CrewAIRunner satisfies AMLAgentRunner at runtime."""
        from aml_copilot.phase3_compare.protocol import AMLAgentRunner
        from aml_copilot.phase3_compare.crewai_runner import CrewAIRunner
        assert isinstance(CrewAIRunner(), AMLAgentRunner)

    def test_crewai_runner_framework_name(self):
        """CrewAIRunner.framework_name is 'crewai'."""
        from aml_copilot.phase3_compare.crewai_runner import CrewAIRunner
        assert CrewAIRunner.framework_name == "crewai"

    # ── DeterministicCrewAILLM unit tests ─────────────────────────────────────

    def test_crewai_llm_inherits_base_llm(self):
        """DeterministicCrewAILLM is a BaseLLM subclass."""
        from crewai.llms.base_llm import BaseLLM
        from aml_copilot.phase3_compare.crewai_runner import DeterministicCrewAILLM
        assert issubclass(DeterministicCrewAILLM, BaseLLM)

    def test_crewai_llm_call_returns_string(self):
        """DeterministicCrewAILLM.call() always returns a str."""
        from aml_copilot.phase3_compare.crewai_runner import DeterministicCrewAILLM
        llm = DeterministicCrewAILLM(model="aml-offline", evidence_json="{}")
        result = llm.call(messages=[])
        assert isinstance(result, str)

    def test_crewai_llm_call_invokes_mock_llm(self):
        """call() output encodes a dict from mock_llm_call (5 required keys)."""
        import json
        from aml_copilot.phase3_compare.crewai_runner import DeterministicCrewAILLM
        from aml_copilot.phase3_compare.mock_llm import MockLLMOutput
        evidence = {"sanctions_hits": [], "rule_firings": [], "anomaly_score": None}
        llm = DeterministicCrewAILLM(
            model="aml-offline", evidence_json=json.dumps(evidence)
        )
        raw = llm.call(messages=[])
        # Strip the ReAct wrapper to get the JSON portion
        json_part = raw.split("Final Answer:")[-1].strip()
        parsed = json.loads(json_part)
        for key in ("disposition", "decision_reason", "reasoning", "confidence", "human_review"):
            assert key in parsed, f"Missing key {key!r} in LLM output"

    def test_crewai_llm_call_output_contains_final_answer(self):
        """call() output contains 'Final Answer:' so CrewAI's executor accepts it."""
        import json
        from aml_copilot.phase3_compare.crewai_runner import DeterministicCrewAILLM
        llm = DeterministicCrewAILLM(model="aml-offline", evidence_json="{}")
        result = llm.call(messages=[])
        assert "Final Answer:" in result

    def test_crewai_no_network_calls(self, monkeypatch):
        """_run_crew() makes no HTTP/network calls during execution."""
        import httpx
        import urllib.request
        from aml_copilot.phase3_compare.crewai_runner import _run_crew

        network_calls: list[str] = []

        def fail_httpx(self_inner, *args, **kwargs):
            network_calls.append(f"httpx.Client.request({args[:1]})")
            raise RuntimeError("Network call intercepted!")

        def fail_httpx_async(self_inner, *args, **kwargs):
            network_calls.append(f"httpx.AsyncClient.request({args[:1]})")
            raise RuntimeError("Network call intercepted!")

        def fail_urllib(*args, **kwargs):
            network_calls.append(f"urllib.urlopen({args[:1]})")
            raise RuntimeError("Network call intercepted!")

        monkeypatch.setattr(httpx.Client, "request", fail_httpx)
        monkeypatch.setattr(httpx.AsyncClient, "request", fail_httpx_async)
        monkeypatch.setattr(urllib.request, "urlopen", fail_urllib)

        evidence = {"sanctions_hits": [], "rule_firings": [], "anomaly_score": None}
        _run_crew(evidence, verbose=False)
        assert network_calls == [], f"Unexpected network calls during _run_crew: {network_calls}"

    def test_crewai_executes_real_crew_tasks(self, monkeypatch):
        """Crew.kickoff() is genuinely called inside _run_crew()."""
        from crewai import Crew
        from aml_copilot.phase3_compare.crewai_runner import _run_crew

        kickoff_calls = []
        original_kickoff = Crew.kickoff

        def spy_kickoff(self, *args, **kwargs):
            kickoff_calls.append(True)
            return original_kickoff(self, *args, **kwargs)

        monkeypatch.setattr(Crew, "kickoff", spy_kickoff)

        evidence = {"sanctions_hits": [], "rule_firings": [], "anomaly_score": None}
        _run_crew(evidence, verbose=False)

        assert len(kickoff_calls) == 1, (
            f"Expected Crew.kickoff to be called once; got {len(kickoff_calls)}"
        )

    # ── Full pipeline invariants (module-scoped fixture) ──────────────────────

    def test_crewai_result_count(self, _crewai_results):
        """CrewAI adapter returns exactly 90 results."""
        assert len(_crewai_results) == 90

    def test_crewai_framework_tag(self, _crewai_results):
        """Every result carries framework == 'crewai'."""
        assert all(r.framework == "crewai" for r in _crewai_results)

    def test_crewai_schema_valid(self, _crewai_results):
        """Every returned object is a valid Phase3CaseResult."""
        from aml_copilot.schemas import Phase3CaseResult
        for r in _crewai_results:
            assert isinstance(r, Phase3CaseResult)

    def test_crewai_unique_case_ids(self, _crewai_results):
        """No duplicate case IDs in the output."""
        ids = [r.case_id for r in _crewai_results]
        assert len(ids) == len(set(ids))

    def test_crewai_dispositions_match_langgraph(self, _crewai_results, _langgraph_results):
        """All 90 CrewAI dispositions are identical to LangGraph dispositions."""
        lg_map = {r.case_id: r.disposition for r in _langgraph_results}
        mismatches = [
            (r.case_id, r.disposition, lg_map.get(r.case_id))
            for r in _crewai_results
            if r.disposition != lg_map.get(r.case_id)
        ]
        assert mismatches == [], (
            f"{len(mismatches)} disposition mismatches between CrewAI and LangGraph: "
            f"{mismatches[:5]}"
        )

    def test_crewai_accuracy_matches_langgraph(self, _crewai_results):
        """CrewAI disposition accuracy equals the committed Phase 2 metric."""
        gold: dict[str, str] = {}
        with open(_EVAL_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    from aml_copilot.schemas import EvalCase
                    c = EvalCase.model_validate_json(line)
                    gold[c.case_id] = c.gold_label

        correct = sum(1 for r in _crewai_results if gold.get(r.case_id) == r.disposition)
        accuracy = correct / len(_crewai_results)
        assert accuracy == pytest.approx(_EXPECTED_P2_ACCURACY, abs=_ACCURACY_TOL), (
            f"CrewAI accuracy {accuracy:.6f} differs from expected {_EXPECTED_P2_ACCURACY:.6f}"
        )

    def test_crewai_delegates_policy_to_mock_llm(self, monkeypatch):
        """Patching mock_llm_call changes CrewAI output — confirms no inline policy in adapter."""
        import json
        import aml_copilot.phase3_compare.crewai_runner as module
        from aml_copilot.phase3_compare.crewai_runner import _run_crew

        def always_clear(evidence):
            return {
                "disposition": "CLEAR",
                "decision_reason": "patched_policy",
                "reasoning": "patched",
                "confidence": 1.0,
                "human_review": False,
            }

        monkeypatch.setattr(module, "mock_llm_call", always_clear)

        # High-sanctions evidence should still CLEAR because mock_llm_call is patched
        evidence = {
            "sanctions_hits": [{"match_score": 0.99}],
            "rule_firings": [],
            "anomaly_score": None,
            "baseline_disposition": "ESCALATE",
            "baseline_reason": "sanctions_or_critical_rule",
        }
        result = _run_crew(evidence, verbose=False)
        assert result["disposition"] == "CLEAR", (
            "CrewAI adapter ignored mock_llm_call patch — inline decision policy suspected"
        )

    def test_crewai_agent_reasoning_populated(self, _crewai_results):
        """Every result has a non-empty agent_reasoning string."""
        empty = [r.case_id for r in _crewai_results if not r.agent_reasoning]
        assert empty == [], f"Empty agent_reasoning for case_ids: {empty}"

    def test_crewai_human_review_matches_langgraph(self, _crewai_results, _langgraph_results):
        """human_review_flagged matches LangGraph for all 90 cases."""
        lg_map = {r.case_id: r.human_review_flagged for r in _langgraph_results}
        mismatches = [
            (r.case_id, r.human_review_flagged, lg_map.get(r.case_id))
            for r in _crewai_results
            if r.human_review_flagged != lg_map.get(r.case_id)
        ]
        assert mismatches == [], (
            f"{len(mismatches)} human_review_flagged mismatches: {mismatches[:5]}"
        )

    def test_crewai_baseline_disposition_populated(self, _crewai_results):
        """Every result has a non-empty baseline_disposition field."""
        missing = [
            r.case_id for r in _crewai_results
            if r.baseline_disposition not in ("ESCALATE", "CLEAR")
        ]
        assert missing == [], f"Missing/invalid baseline_disposition for: {missing}"


# ══════════════════════════════════════════════════════════════════════════════
# 9. OpenAI Agents SDK adapter — unit, integration, parity (M4, tests 54–78)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def _openai_agents_results(_fixture_files_available: bool) -> list:
    """Run the OpenAI Agents SDK adapter once; share results across all M4 tests."""
    if not _fixture_files_available:
        pytest.skip("Fixture files not available")
    from aml_copilot.phase3_compare.openai_agents_runner import OpenAIAgentsRunner
    return OpenAIAgentsRunner().run(_EVAL_PATH, _BASELINE_PATH)


class TestOpenAIAgentsRunner:
    """Tests 54–78: Phase 3 M4 OpenAI Agents SDK adapter."""
    pytestmark = [pytest.mark.integration, pytest.mark.compare, pytest.mark.slow]

    # ── SDK imports and protocol ───────────────────────────────────────────────

    def test_openai_agents_sdk_imports_cleanly(self):
        """All required SDK symbols and the runner import without credentials."""
        from agents import Agent, Runner, function_tool  # noqa: F401
        from aml_copilot.phase3_compare.openai_agents_runner import OpenAIAgentsRunner  # noqa: F401

    def test_openai_agents_runner_protocol(self):
        """OpenAIAgentsRunner satisfies AMLAgentRunner at runtime."""
        from aml_copilot.phase3_compare.protocol import AMLAgentRunner
        from aml_copilot.phase3_compare.openai_agents_runner import OpenAIAgentsRunner
        assert isinstance(OpenAIAgentsRunner(), AMLAgentRunner)

    def test_openai_agents_framework_name(self):
        """OpenAIAgentsRunner.framework_name is 'openai_agents'."""
        from aml_copilot.phase3_compare.openai_agents_runner import OpenAIAgentsRunner
        assert OpenAIAgentsRunner.framework_name == "openai_agents"

    # ── Full pipeline invariants (module-scoped fixture) ──────────────────────

    def test_openai_agents_runner_returns_90(self, _openai_agents_results):
        """Adapter returns exactly 90 results."""
        assert len(_openai_agents_results) == 90

    def test_openai_agents_framework_tag(self, _openai_agents_results):
        """Every result carries framework == 'openai_agents'."""
        assert all(r.framework == "openai_agents" for r in _openai_agents_results)

    def test_openai_agents_schema_valid(self, _openai_agents_results):
        """Every returned object is a valid Phase3CaseResult."""
        from aml_copilot.schemas import Phase3CaseResult
        for r in _openai_agents_results:
            assert isinstance(r, Phase3CaseResult)

    def test_openai_agents_unique_case_ids(self, _openai_agents_results):
        """No duplicate case IDs in the output."""
        ids = [r.case_id for r in _openai_agents_results]
        assert len(ids) == len(set(ids))

    def test_openai_agents_dispositions_match_langgraph(
        self, _openai_agents_results, _langgraph_results
    ):
        """All 90 dispositions are identical to LangGraph."""
        lg_map = {r.case_id: r.disposition for r in _langgraph_results}
        mismatches = [
            (r.case_id, r.disposition, lg_map.get(r.case_id))
            for r in _openai_agents_results
            if r.disposition != lg_map.get(r.case_id)
        ]
        assert mismatches == [], (
            f"{len(mismatches)} disposition mismatches vs LangGraph: {mismatches[:5]}"
        )

    def test_openai_agents_dispositions_match_crewai(
        self, _openai_agents_results, _crewai_results
    ):
        """All 90 dispositions are identical to CrewAI."""
        ca_map = {r.case_id: r.disposition for r in _crewai_results}
        mismatches = [
            (r.case_id, r.disposition, ca_map.get(r.case_id))
            for r in _openai_agents_results
            if r.disposition != ca_map.get(r.case_id)
        ]
        assert mismatches == [], (
            f"{len(mismatches)} disposition mismatches vs CrewAI: {mismatches[:5]}"
        )

    def test_openai_agents_policy_fields_match_langgraph(
        self, _openai_agents_results, _langgraph_results
    ):
        """All policy-output fields match LangGraph for all 90 cases."""
        lg_map = {r.case_id: r for r in _langgraph_results}
        policy_fields = [
            "disposition", "decision_reason", "agent_reasoning",
            "agent_override", "baseline_disposition", "human_review_flagged",
            "tokens_used", "cost_usd",
        ]
        mismatches = []
        for r in _openai_agents_results:
            lg = lg_map.get(r.case_id)
            if lg is None:
                mismatches.append((r.case_id, "missing from LangGraph"))
                continue
            for field in policy_fields:
                oa_val = getattr(r, field)
                lg_val = getattr(lg, field)
                if oa_val != lg_val:
                    mismatches.append((r.case_id, field, oa_val, lg_val))
        assert mismatches == [], (
            f"Policy field mismatches vs LangGraph: {mismatches[:5]}"
        )

    def test_openai_agents_evidence_fields_match(
        self, _openai_agents_results, _langgraph_results
    ):
        """Evidence fields (sanctions_hits, rule_firings, anomaly_score) match LangGraph."""
        lg_map = {r.case_id: r for r in _langgraph_results}
        evidence_fields = ["sanctions_hits", "rule_firings", "anomaly_score"]
        mismatches = []
        for r in _openai_agents_results:
            lg = lg_map.get(r.case_id)
            if lg is None:
                continue
            for field in evidence_fields:
                if getattr(r, field) != getattr(lg, field):
                    mismatches.append((r.case_id, field))
        assert mismatches == [], (
            f"Evidence field mismatches vs LangGraph: {mismatches[:5]}"
        )

    def test_openai_agents_accuracy_matches(self, _openai_agents_results):
        """Disposition accuracy equals the committed Phase 2 metric."""
        gold: dict[str, str] = {}
        with open(_EVAL_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    from aml_copilot.schemas import EvalCase
                    c = EvalCase.model_validate_json(line)
                    gold[c.case_id] = c.gold_label

        correct = sum(1 for r in _openai_agents_results if gold.get(r.case_id) == r.disposition)
        accuracy = correct / len(_openai_agents_results)
        assert accuracy == pytest.approx(_EXPECTED_P2_ACCURACY, abs=1e-10), (
            f"OpenAI Agents accuracy {accuracy:.10f} differs from "
            f"expected {_EXPECTED_P2_ACCURACY:.10f}"
        )

    def test_openai_agents_zero_cost(self, _openai_agents_results):
        """Every result has tokens_used == 0 and cost_usd == 0.0."""
        bad = [
            (r.case_id, r.tokens_used, r.cost_usd)
            for r in _openai_agents_results
            if r.tokens_used != 0 or r.cost_usd != 0.0
        ]
        assert bad == [], f"Non-zero cost/token entries: {bad}"

    def test_openai_agents_no_network_calls(self, monkeypatch, _require_fixture_files):
        """Running one case makes zero HTTP calls."""
        import httpx
        import urllib.request

        network_calls: list[str] = []

        def fail_httpx(self_inner, *args, **kwargs):
            network_calls.append(f"httpx.Client.request({args[:1]})")
            raise RuntimeError("Network call intercepted!")

        def fail_httpx_async(self_inner, *args, **kwargs):
            network_calls.append(f"httpx.AsyncClient.request({args[:1]})")
            raise RuntimeError("Network call intercepted!")

        def fail_urllib(*args, **kwargs):
            network_calls.append(f"urllib.urlopen({args[:1]})")
            raise RuntimeError("Network call intercepted!")

        monkeypatch.setattr(httpx.Client, "request", fail_httpx)
        monkeypatch.setattr(httpx.AsyncClient, "request", fail_httpx_async)
        monkeypatch.setattr(urllib.request, "urlopen", fail_urllib)

        from aml_copilot.phase3_compare.openai_agents_runner import (
            OpenAIAgentsRunner,
        )

        runner = OpenAIAgentsRunner()
        results = runner.run(_EVAL_PATH, _BASELINE_PATH)
        assert len(results) == 90
        assert network_calls == [], (
            f"Unexpected network calls: {network_calls}"
        )

    def test_openai_agents_no_api_key_required(self, monkeypatch, _require_fixture_files):
        """Runner succeeds with OPENAI_API_KEY removed from environment."""
        import os
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from aml_copilot.phase3_compare.openai_agents_runner import (
            OpenAIAgentsRunner,
        )

        runner = OpenAIAgentsRunner()
        results = runner.run(_EVAL_PATH, _BASELINE_PATH)
        assert len(results) == 90

    def test_openai_agents_environment_unchanged(
        self, monkeypatch, _require_fixture_files
    ):
        """Run does not mutate any OpenAI environment variables."""
        import os

        tracked_vars = [
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_ORG_ID",
            "OPENAI_PROJECT_ID",
            "OPENAI_AGENTS_DISABLE_TRACING",
        ]
        before = {v: os.environ.get(v) for v in tracked_vars}

        from aml_copilot.phase3_compare.openai_agents_runner import (
            OpenAIAgentsRunner,
        )

        OpenAIAgentsRunner().run(_EVAL_PATH, _BASELINE_PATH)
        after = {v: os.environ.get(v) for v in tracked_vars}
        assert before == after, f"Environment mutated: {before} -> {after}"

    def test_openai_agents_runner_actually_invoked(self, monkeypatch, _require_fixture_files):
        """Runner.run_sync is genuinely called during a full run."""
        from agents import Runner
        import aml_copilot.phase3_compare.openai_agents_runner as module

        run_sync_count = {"n": 0}
        original_run_sync = Runner.run_sync

        def spy_run_sync(agent, input, **kwargs):
            run_sync_count["n"] += 1
            return original_run_sync(agent, input, **kwargs)

        monkeypatch.setattr(Runner, "run_sync", staticmethod(spy_run_sync))

        module.OpenAIAgentsRunner().run(_EVAL_PATH, _BASELINE_PATH)
        assert run_sync_count["n"] == 90, (
            f"Expected 90 Runner.run_sync calls; got {run_sync_count['n']}"
        )

    def test_openai_agents_function_tools_are_invoked(self, monkeypatch, _require_fixture_files):
        """Both parse_evidence and decide_disposition execute during one case run."""
        import json
        import aml_copilot.phase3_compare.openai_agents_runner as module

        invoked: list[str] = []
        orig_parse = module.parse_evidence.on_invoke_tool
        orig_decide = module.decide_disposition.on_invoke_tool

        async def spy_parse(ctx, args):
            invoked.append("parse_evidence")
            return await orig_parse(ctx, args)

        async def spy_decide(ctx, args):
            invoked.append("decide_disposition")
            return await orig_decide(ctx, args)

        monkeypatch.setattr(module.parse_evidence, "on_invoke_tool", spy_parse)
        monkeypatch.setattr(module.decide_disposition, "on_invoke_tool", spy_decide)

        from aml_copilot.phase3_compare.openai_agents_runner import (
            _run_agent,
            _build_case_data,
            _create_agent,
            _load_baseline_results,
        )
        from agents import RunConfig

        baseline_map = _load_baseline_results(_BASELINE_PATH)
        first_baseline = next(iter(baseline_map.values()))
        case_data = _build_case_data(first_baseline)
        agent = _create_agent()
        _run_agent(agent, json.dumps(case_data), RunConfig(tracing_disabled=True))

        assert "parse_evidence" in invoked, "parse_evidence was not invoked"
        assert "decide_disposition" in invoked, "decide_disposition was not invoked"

    def test_openai_agents_structured_output_used(self, _openai_agents_results):
        """final_output for every case is an OpenAIAgentDecision Pydantic instance."""
        import json
        from aml_copilot.phase3_compare.openai_agents_runner import (
            OpenAIAgentDecision,
            _run_agent,
            _build_case_data,
            _create_agent,
            _load_baseline_results,
        )
        from agents import RunConfig
        baseline_map = _load_baseline_results(_BASELINE_PATH)
        first_baseline = next(iter(baseline_map.values()))
        case_data = _build_case_data(first_baseline)
        agent = _create_agent()
        result = _run_agent(agent, json.dumps(case_data), RunConfig(tracing_disabled=True))
        assert isinstance(result.final_output, OpenAIAgentDecision), (
            f"Expected OpenAIAgentDecision, got {type(result.final_output).__name__}"
        )

    def test_openai_agents_deterministic(self, _require_fixture_files):
        """Same input produces identical non-latency output on two successive runs."""
        import json
        from aml_copilot.phase3_compare.openai_agents_runner import (
            _run_agent,
            _build_case_data,
            _create_agent,
            _load_baseline_results,
        )
        from agents import RunConfig
        baseline_map = _load_baseline_results(_BASELINE_PATH)
        first_baseline = next(iter(baseline_map.values()))
        case_data = _build_case_data(first_baseline)
        agent = _create_agent()
        run_config = RunConfig(tracing_disabled=True)
        input_json = json.dumps(case_data)

        r1 = _run_agent(agent, input_json, run_config).final_output
        r2 = _run_agent(agent, input_json, run_config).final_output

        assert r1.model_dump() == r2.model_dump(), (
            f"Non-deterministic output: {r1.model_dump()} != {r2.model_dump()}"
        )

    def test_openai_agents_invalid_join_raises(self, _require_fixture_files):
        """Missing baseline result raises RuntimeError naming the case_id."""
        import tempfile, json
        from aml_copilot.phase3_compare.openai_agents_runner import OpenAIAgentsRunner
        from aml_copilot.schemas import EvalCase

        # Build eval with one case that has no matching baseline
        with open(_EVAL_PATH, encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        first_case = EvalCase.model_validate_json(lines[0])

        # Create a baseline file with only a different account
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            empty_baseline = f.name

        import pathlib
        pathlib.Path(empty_baseline).write_text("", encoding="utf-8")

        with pytest.raises(RuntimeError, match=first_case.case_id):
            OpenAIAgentsRunner().run(_EVAL_PATH, pathlib.Path(empty_baseline))

        pathlib.Path(empty_baseline).unlink(missing_ok=True)

    def test_openai_agents_duplicate_case_id_raises(self):
        """Duplicate case IDs in results trigger a RuntimeError."""
        import json
        from aml_copilot.phase3_compare.openai_agents_runner import _validate
        from aml_copilot.schemas import Phase3CaseResult, SanctionsHit
        from datetime import datetime, timezone

        # Build two results with the same case_id
        dummy = Phase3CaseResult(
            framework="openai_agents",
            case_id="DUPLICATE-001",
            account_id="ACC-001",
            disposition="CLEAR",
            decision_reason="clear",
            sanctions_hits=[],
            rule_firings=[],
            anomaly_score=None,
            latency_ms=1.0,
            agent_reasoning="no risk",
            agent_override=False,
            baseline_disposition="CLEAR",
            human_review_flagged=False,
        )
        # Fill to 90 with distinct IDs except one duplicate pair
        results = []
        for i in range(89):
            results.append(dummy.model_copy(update={"case_id": f"CASE-{i:03d}"}))
        results.append(dummy.model_copy(update={"case_id": "CASE-000"}))  # duplicate

        with pytest.raises(RuntimeError, match="duplicate"):
            _validate(results)

    def test_openai_agents_framework_failure_never_clears(self, monkeypatch, _require_fixture_files):
        """SDK execution failure raises RuntimeError — never returns a fabricated CLEAR."""
        from agents import Runner
        from aml_copilot.phase3_compare.openai_agents_runner import OpenAIAgentsRunner

        def fail_run_sync(*args, **kwargs):
            raise RuntimeError("Simulated SDK failure")

        monkeypatch.setattr(Runner, "run_sync", staticmethod(fail_run_sync))

        with pytest.raises(RuntimeError):
            OpenAIAgentsRunner().run(_EVAL_PATH, _BASELINE_PATH)

    # ── Three-framework aggregate tests ───────────────────────────────────────

    def test_all_three_frameworks_agree(
        self, _openai_agents_results, _langgraph_results, _crewai_results
    ):
        """All 90 dispositions are identical across LangGraph, CrewAI, and OpenAI Agents."""
        lg_map = {r.case_id: r.disposition for r in _langgraph_results}
        ca_map = {r.case_id: r.disposition for r in _crewai_results}
        diffs = []
        for r in _openai_agents_results:
            lg = lg_map.get(r.case_id)
            ca = ca_map.get(r.case_id)
            if r.disposition != lg or r.disposition != ca:
                diffs.append((r.case_id, r.disposition, lg, ca))
        assert diffs == [], f"Three-framework disagreements: {diffs[:5]}"

    def test_all_three_zero_cost(
        self, _openai_agents_results, _langgraph_results, _crewai_results
    ):
        """Every result from every framework has tokens_used == 0 and cost_usd == 0.0."""
        all_results = (
            list(_langgraph_results)
            + list(_crewai_results)
            + list(_openai_agents_results)
        )
        bad = [
            (r.framework, r.case_id, r.tokens_used, r.cost_usd)
            for r in all_results
            if r.tokens_used != 0 or r.cost_usd != 0.0
        ]
        assert bad == [], f"Non-zero cost/token entries: {bad}"


# ══════════════════════════════════════════════════════════════════════════════
# 10. Shared helpers (_shared.py) — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSharedHelpers:
    """Tests for phase3_compare._shared: EXPECTED_EVAL_SIZE, build_evidence, validate."""

    def test_expected_eval_size_is_90(self):
        """EXPECTED_EVAL_SIZE equals 90 — the only permitted eval set size."""
        from aml_copilot.phase3_compare._shared import EXPECTED_EVAL_SIZE
        assert EXPECTED_EVAL_SIZE == 90

    def test_build_evidence_returns_five_keys(self):
        """build_evidence() returns exactly the 5 keys consumed by mock_llm_call."""
        from aml_copilot.phase3_compare._shared import build_evidence
        from aml_copilot.schemas import CaseResult

        baseline = CaseResult(
            case_id="TEST-001",
            account_id="ACC-001",
            disposition="CLEAR",
            decision_reason="clear",
            sanctions_hits=[],
            rule_firings=[],
            anomaly_score=None,
            latency_ms=1.0,
        )
        evidence = build_evidence(baseline)
        assert set(evidence.keys()) == {
            "sanctions_hits",
            "rule_firings",
            "anomaly_score",
            "baseline_disposition",
            "baseline_reason",
        }

    def test_build_evidence_maps_fields_correctly(self):
        """build_evidence() copies the right fields from CaseResult."""
        from aml_copilot.phase3_compare._shared import build_evidence
        from aml_copilot.schemas import CaseResult

        baseline = CaseResult(
            case_id="TEST-002",
            account_id="ACC-002",
            disposition="ESCALATE",
            decision_reason="sanctions_or_critical_rule",
            sanctions_hits=[],
            rule_firings=[],
            anomaly_score=None,
            latency_ms=2.0,
        )
        ev = build_evidence(baseline)
        assert ev["baseline_disposition"] == "ESCALATE"
        assert ev["baseline_reason"] == "sanctions_or_critical_rule"
        assert ev["sanctions_hits"] == []
        assert ev["rule_firings"] == []
        assert ev["anomaly_score"] is None

    def test_validate_wrong_count_raises(self):
        """validate_phase3_results raises RuntimeError when result count != eval count."""
        from aml_copilot.phase3_compare._shared import validate_phase3_results, EXPECTED_EVAL_SIZE
        from aml_copilot.schemas import EvalCase, Phase3CaseResult

        def _minimal_result(case_id: str) -> Phase3CaseResult:
            return Phase3CaseResult(
                framework="langgraph",
                case_id=case_id,
                account_id="ACC-001",
                disposition="CLEAR",
                decision_reason="clear",
                sanctions_hits=[],
                rule_firings=[],
                anomaly_score=None,
                latency_ms=1.0,
                agent_reasoning="test",
                agent_override=False,
                baseline_disposition="CLEAR",
                human_review_flagged=False,
            )

        def _minimal_case(case_id: str) -> EvalCase:
            return EvalCase(
                case_id=case_id,
                account_id="ACC-001",
                gold_label="CLEAR",
                case_type="ibm_labeled",
                relevant_txn_ids=[],
                notes="",
            )

        # 89 results vs 90 cases — should raise
        results = [_minimal_result(f"CASE-{i:03d}") for i in range(89)]
        eval_cases = [_minimal_case(f"CASE-{i:03d}") for i in range(EXPECTED_EVAL_SIZE)]
        with pytest.raises(RuntimeError):
            validate_phase3_results(results, eval_cases, "langgraph")

    def test_validate_duplicate_case_id_raises(self):
        """validate_phase3_results raises RuntimeError on duplicate case_ids."""
        from aml_copilot.phase3_compare._shared import validate_phase3_results, EXPECTED_EVAL_SIZE
        from aml_copilot.schemas import EvalCase, Phase3CaseResult

        def _r(case_id: str) -> Phase3CaseResult:
            return Phase3CaseResult(
                framework="crewai",
                case_id=case_id,
                account_id="ACC-001",
                disposition="CLEAR",
                decision_reason="clear",
                sanctions_hits=[],
                rule_firings=[],
                anomaly_score=None,
                latency_ms=1.0,
                agent_reasoning="test",
                agent_override=False,
                baseline_disposition="CLEAR",
                human_review_flagged=False,
            )

        def _c(case_id: str) -> EvalCase:
            return EvalCase(
                case_id=case_id,
                account_id="ACC-001",
                gold_label="CLEAR",
                case_type="ibm_labeled",
                relevant_txn_ids=[],
                notes="",
            )

        # 89 unique + 1 duplicate == 90 results, but one case_id appears twice
        results = [_r(f"CASE-{i:03d}") for i in range(89)] + [_r("CASE-000")]
        eval_cases = [_c(f"CASE-{i:03d}") for i in range(EXPECTED_EVAL_SIZE)]
        with pytest.raises(RuntimeError, match="duplicate"):
            validate_phase3_results(results, eval_cases, "crewai")

    def test_validate_wrong_framework_tag_raises(self):
        """validate_phase3_results raises RuntimeError when a result has the wrong framework tag."""
        from aml_copilot.phase3_compare._shared import validate_phase3_results, EXPECTED_EVAL_SIZE
        from aml_copilot.schemas import EvalCase, Phase3CaseResult

        def _r(case_id: str, fw: str) -> Phase3CaseResult:
            return Phase3CaseResult(
                framework=fw,
                case_id=case_id,
                account_id="ACC-001",
                disposition="CLEAR",
                decision_reason="clear",
                sanctions_hits=[],
                rule_firings=[],
                anomaly_score=None,
                latency_ms=1.0,
                agent_reasoning="test",
                agent_override=False,
                baseline_disposition="CLEAR",
                human_review_flagged=False,
            )

        def _c(case_id: str) -> EvalCase:
            return EvalCase(
                case_id=case_id,
                account_id="ACC-001",
                gold_label="CLEAR",
                case_type="ibm_labeled",
                relevant_txn_ids=[],
                notes="",
            )

        results = [_r(f"CASE-{i:03d}", "langgraph") for i in range(89)] + \
                  [_r("CASE-089", "crewai")]  # wrong tag
        eval_cases = [_c(f"CASE-{i:03d}") for i in range(EXPECTED_EVAL_SIZE)]
        with pytest.raises(RuntimeError, match="wrong framework tag"):
            validate_phase3_results(results, eval_cases, "langgraph")


# ══════════════════════════════════════════════════════════════════════════════
# 11. Evidence parity — all three adapters produce identical evidence dicts
# ══════════════════════════════════════════════════════════════════════════════

class TestEvidenceParity:
    pytestmark = [pytest.mark.integration, pytest.mark.compare]
    """Tests asserting all 3 adapters pass identical evidence to mock_llm_call."""

    def test_build_evidence_langgraph_crewai_identical(self, _langgraph_results, _crewai_results):
        """LangGraph and CrewAI carry identical sanctions_hits, rule_firings, anomaly_score."""
        lg_map = {r.case_id: r for r in _langgraph_results}
        mismatches = []
        for ca in _crewai_results:
            lg = lg_map.get(ca.case_id)
            if lg is None:
                mismatches.append((ca.case_id, "missing from LangGraph"))
                continue
            for field in ("sanctions_hits", "rule_firings", "anomaly_score"):
                if getattr(ca, field) != getattr(lg, field):
                    mismatches.append((ca.case_id, field))
        assert mismatches == [], f"Evidence mismatch (CrewAI vs LangGraph): {mismatches[:5]}"

    def test_build_evidence_openai_langgraph_identical(
        self, _openai_agents_results, _langgraph_results
    ):
        """OpenAI Agents and LangGraph carry identical sanctions_hits, rule_firings, anomaly_score."""
        lg_map = {r.case_id: r for r in _langgraph_results}
        mismatches = []
        for oa in _openai_agents_results:
            lg = lg_map.get(oa.case_id)
            if lg is None:
                mismatches.append((oa.case_id, "missing from LangGraph"))
                continue
            for field in ("sanctions_hits", "rule_firings", "anomaly_score"):
                if getattr(oa, field) != getattr(lg, field):
                    mismatches.append((oa.case_id, field))
        assert mismatches == [], (
            f"Evidence mismatch (OpenAI Agents vs LangGraph): {mismatches[:5]}"
        )

    def test_build_evidence_unit_produces_identical_output(self):
        """build_evidence() called on the same CaseResult produces the same dict every time."""
        from aml_copilot.phase3_compare._shared import build_evidence
        from aml_copilot.schemas import CaseResult

        baseline = CaseResult(
            case_id="PARITY-001",
            account_id="ACC-001",
            disposition="ESCALATE",
            decision_reason="sanctions_or_critical_rule",
            sanctions_hits=[],
            rule_firings=[],
            anomaly_score=None,
            latency_ms=1.0,
        )
        e1 = build_evidence(baseline)
        e2 = build_evidence(baseline)
        assert e1 == e2


# ══════════════════════════════════════════════════════════════════════════════
# 12. Policy boundaries through adapter execution paths (behavioral)
# ══════════════════════════════════════════════════════════════════════════════

class TestPolicyBoundariesThroughAdapters:
    pytestmark = [pytest.mark.integration, pytest.mark.compare]
    """Behavioral tests replacing source inspection: adapters delegate all policy to mock_llm_call."""

    def test_langgraph_delegates_policy_to_mock_llm(self, monkeypatch):
        """Patching mock_llm_call changes LangGraph output — confirms no inline policy."""
        import aml_copilot.phase2_eval.run_langgraph_eval as module

        def always_clear(evidence):
            return {
                "disposition": "CLEAR",
                "decision_reason": "patched_policy",
                "reasoning": "patched",
                "confidence": 1.0,
                "human_review": False,
            }

        monkeypatch.setattr(module, "mock_llm_call", always_clear)

        from aml_copilot.phase2_eval.run_langgraph_eval import (
            build_graph,
            AMLAgentState,
        )
        from aml_copilot.schemas import CaseResult

        baseline = CaseResult(
            case_id="POLICY-001",
            account_id="ACC-001",
            disposition="ESCALATE",
            decision_reason="sanctions_or_critical_rule",
            sanctions_hits=[],
            rule_firings=[],
            anomaly_score=None,
            latency_ms=1.0,
        )
        graph = build_graph()
        initial_state: AMLAgentState = {
            "case_id": "POLICY-001",
            "account_id": "ACC-001",
            "case_type": "ibm_labeled",
            "notes": "",
            "baseline_result_json": baseline.model_dump_json(),
            "evidence": {},
            "agent_disposition": "",
            "agent_decision_reason": "",
            "agent_reasoning": "",
            "agent_confidence": 0.0,
            "human_review_flagged": False,
            "tokens_used": 0,
            "cost_usd": 0.0,
        }
        final = graph.invoke(initial_state)
        assert final["agent_disposition"] == "CLEAR", (
            "LangGraph adapter ignored mock_llm_call patch — inline policy suspected"
        )

    def test_openai_agents_delegates_policy_to_mock_llm(self, monkeypatch):
        """Patching mock_llm_call changes OpenAI Agents output — confirms no inline policy."""
        import json
        import aml_copilot.phase3_compare.openai_agents_runner as module

        def always_clear(evidence):
            return {
                "disposition": "CLEAR",
                "decision_reason": "patched_policy",
                "reasoning": "patched",
                "confidence": 1.0,
                "human_review": False,
            }

        monkeypatch.setattr(module, "mock_llm_call", always_clear)

        from aml_copilot.phase3_compare.openai_agents_runner import (
            _run_agent,
            _create_agent,
        )
        from agents import RunConfig

        evidence = {
            "sanctions_hits": [{"match_score": 0.99}],
            "rule_firings": [],
            "anomaly_score": None,
            "baseline_disposition": "ESCALATE",
            "baseline_reason": "sanctions_or_critical_rule",
        }
        agent = _create_agent()
        sdk_result = _run_agent(agent, json.dumps(evidence), RunConfig(tracing_disabled=True))
        assert sdk_result.final_output.disposition == "CLEAR", (
            "OpenAI Agents adapter ignored mock_llm_call patch — inline policy suspected"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 13. Latency methodology — timers wrap only framework invocation
# ══════════════════════════════════════════════════════════════════════════════

class TestLatencyMethodology:
    pytestmark = [pytest.mark.integration, pytest.mark.compare, pytest.mark.slow]
    """Verify that per-case latency measures only the framework invocation call."""

    def test_crewai_kickoff_is_timed(self, monkeypatch):
        """CrewAIRunner.run() records positive latency_ms sourced from crew.kickoff()."""
        from crewai import Crew
        import aml_copilot.phase3_compare.crewai_runner as module

        kickoff_durations: list[float] = []
        original_kickoff = Crew.kickoff
        import time as _time

        def timed_kickoff(self_inner, *args, **kwargs):
            t0 = _time.perf_counter()
            result = original_kickoff(self_inner, *args, **kwargs)
            kickoff_durations.append((_time.perf_counter() - t0) * 1000)
            return result

        monkeypatch.setattr(Crew, "kickoff", timed_kickoff)

        from aml_copilot.phase3_compare.crewai_runner import _run_crew
        evidence = {"sanctions_hits": [], "rule_firings": [], "anomaly_score": None}
        _run_crew(evidence, verbose=False)
        assert len(kickoff_durations) == 1, "Expected exactly one Crew.kickoff call"

    def test_openai_agents_run_sync_is_timed(self, monkeypatch):
        """OpenAIAgentsRunner records positive latency_ms sourced from Runner.run_sync()."""
        from agents import Runner
        import aml_copilot.phase3_compare.openai_agents_runner as module

        run_sync_called = {"count": 0}
        original_run_sync = Runner.run_sync

        def spy_run_sync(agent, input, **kwargs):
            run_sync_called["count"] += 1
            return original_run_sync(agent, input, **kwargs)

        monkeypatch.setattr(Runner, "run_sync", staticmethod(spy_run_sync))

        import json
        from aml_copilot.phase3_compare.openai_agents_runner import (
            _run_agent,
            _create_agent,
        )
        from agents import RunConfig

        evidence = {
            "sanctions_hits": [],
            "rule_firings": [],
            "anomaly_score": None,
            "baseline_disposition": "CLEAR",
            "baseline_reason": "clear",
        }
        agent = _create_agent()
        _run_agent(agent, json.dumps(evidence), RunConfig(tracing_disabled=True))
        assert run_sync_called["count"] == 1, "Expected exactly one Runner.run_sync call"

    def test_crewai_latency_positive_in_full_run(self, _crewai_results):
        """Every CrewAI case result has latency_ms > 0."""
        non_positive = [r.case_id for r in _crewai_results if r.latency_ms <= 0]
        assert non_positive == [], f"Zero/negative latency in cases: {non_positive}"

    def test_openai_agents_latency_positive_in_full_run(self, _openai_agents_results):
        """Every OpenAI Agents case result has latency_ms > 0."""
        non_positive = [r.case_id for r in _openai_agents_results if r.latency_ms <= 0]
        assert non_positive == [], f"Zero/negative latency in cases: {non_positive}"


# ══════════════════════════════════════════════════════════════════════════════
# M5 helpers — shared synthetic data builders
# ══════════════════════════════════════════════════════════════════════════════

def _m5_result(
    case_id: str,
    framework: str = "langgraph",
    disposition: str = "CLEAR",
    decision_reason: str = "clear",
    agent_reasoning: str = "No significant risk.",
    human_review_flagged: bool = False,
    agent_override: bool = False,
    baseline_disposition: str = "CLEAR",
    latency_ms: float = 1.0,
    tokens_used: int = 0,
    cost_usd: float = 0.0,
) -> "Phase3CaseResult":
    from aml_copilot.schemas import Phase3CaseResult
    return Phase3CaseResult(
        framework=framework,
        case_id=case_id,
        account_id=f"ACC-{case_id}",
        disposition=disposition,  # type: ignore[arg-type]
        decision_reason=decision_reason,
        sanctions_hits=[],
        rule_firings=[],
        anomaly_score=None,
        latency_ms=latency_ms,
        agent_reasoning=agent_reasoning,
        agent_override=agent_override,
        baseline_disposition=baseline_disposition,  # type: ignore[arg-type]
        human_review_flagged=human_review_flagged,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
    )


def _m5_eval_case(
    case_id: str,
    gold_label: str = "CLEAR",
    case_type: str = "ibm_labeled",
    severity_band: int | None = None,
) -> "EvalCase":
    from aml_copilot.schemas import EvalCase
    return EvalCase(
        case_id=case_id,
        account_id=f"ACC-{case_id}",
        gold_label=gold_label,  # type: ignore[arg-type]
        case_type=case_type,  # type: ignore[arg-type]
        severity_band=severity_band,  # type: ignore[arg-type]
        relevant_txn_ids=[],
        notes="",
    )


def _m5_results_90(framework: str = "langgraph") -> "list[Phase3CaseResult]":
    """Build 90 synthetic Phase3CaseResult rows matching the known accuracy."""
    from aml_copilot.phase3_compare._shared import EXPECTED_EVAL_SIZE
    results = []
    for i in range(EXPECTED_EVAL_SIZE):
        results.append(_m5_result(f"CASE-{i:03d}", framework=framework))
    return results


def _m5_eval_cases_90() -> "list[EvalCase]":
    from aml_copilot.phase3_compare._shared import EXPECTED_EVAL_SIZE
    return [_m5_eval_case(f"CASE-{i:03d}") for i in range(EXPECTED_EVAL_SIZE)]


# ══════════════════════════════════════════════════════════════════════════════
# 14. M5 Registry and schema tests (no fixture files)
# ══════════════════════════════════════════════════════════════════════════════

class TestM5RegistryAndSchema:
    """Tests for RUNNER_REGISTRY structure, LOC, versions, and schema new fields."""

    def test_registry_has_three_entries(self):
        """RUNNER_REGISTRY contains exactly 3 registered runners."""
        from aml_copilot.phase3_compare.run_comparison import RUNNER_REGISTRY
        assert len(RUNNER_REGISTRY) == 3

    def test_registry_no_duplicate_framework_names(self):
        """No two registry entries share a framework_name."""
        from aml_copilot.phase3_compare.run_comparison import RUNNER_REGISTRY
        names = [cls.framework_name for cls, _ in RUNNER_REGISTRY]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"

    def test_registry_all_satisfy_protocol(self):
        """Every registered runner class satisfies AMLAgentRunner at runtime."""
        from aml_copilot.phase3_compare.run_comparison import RUNNER_REGISTRY
        from aml_copilot.phase3_compare.protocol import AMLAgentRunner
        for runner_cls, _ in RUNNER_REGISTRY:
            assert isinstance(runner_cls(), AMLAgentRunner), (
                f"{runner_cls.__name__} does not satisfy AMLAgentRunner"
            )

    def test_registry_order_is_deterministic(self):
        """Registry order is langgraph → crewai → openai_agents."""
        from aml_copilot.phase3_compare.run_comparison import RUNNER_REGISTRY
        names = [cls.framework_name for cls, _ in RUNNER_REGISTRY]
        assert names == ["langgraph", "crewai", "openai_agents"]

    def test_registry_runner_files_exist(self):
        """Every runner file path in the registry points to an existing file."""
        from aml_copilot.phase3_compare.run_comparison import RUNNER_REGISTRY
        for cls, runner_file in RUNNER_REGISTRY:
            assert runner_file.exists(), (
                f"Runner file not found for {cls.framework_name}: {runner_file}"
            )

    def test_loc_computation_non_blank(self, tmp_path):
        """count_loc counts non-blank lines only."""
        from aml_copilot.phase3_compare.metrics import count_loc
        f = tmp_path / "test.py"
        f.write_text("line1\n\nline3\n   \nline5\n", encoding="utf-8")
        assert count_loc(f) == 3  # blank lines and whitespace-only lines excluded

    def test_loc_computation_real_runner(self):
        """count_loc on a real runner file returns a positive integer."""
        from aml_copilot.phase3_compare.metrics import count_loc
        from aml_copilot.phase3_compare.run_comparison import RUNNER_REGISTRY
        for _, runner_file in RUNNER_REGISTRY:
            loc = count_loc(runner_file)
            assert loc > 0, f"LOC is 0 for {runner_file}"
            assert loc < 2000, f"LOC suspiciously large for {runner_file}: {loc}"

    def test_framework_versions_has_python_key(self):
        """get_framework_versions() returns a dict containing 'python'."""
        from aml_copilot.phase3_compare.metrics import get_framework_versions
        v = get_framework_versions()
        assert "python" in v
        assert isinstance(v["python"], str)
        assert v["python"].count(".") >= 1

    def test_framework_versions_has_framework_keys(self):
        """get_framework_versions() returns entries for all three Phase 3 frameworks."""
        from aml_copilot.phase3_compare.metrics import get_framework_versions
        v = get_framework_versions()
        for key in ("langgraph", "crewai", "openai_agents"):
            assert key in v, f"Missing version key: {key}"

    def test_phase3_framework_metrics_new_fields_have_defaults(self):
        """New Phase3FrameworkMetrics fields (average_latency_ms etc.) have correct defaults."""
        from aml_copilot.schemas import Phase3FrameworkMetrics
        m = Phase3FrameworkMetrics(
            framework="test",
            disposition_accuracy=1.0,
            false_clear_rate_weighted=0.0,
            override_rate=0.0,
            human_review_rate=0.0,
            latency_p50_ms=1.0,
            latency_p95_ms=2.0,
            loc=100,
            total_cost_usd=0.0,
            eval_size=90,
        )
        assert m.average_latency_ms == 0.0
        assert m.minimum_latency_ms == 0.0
        assert m.maximum_latency_ms == 0.0
        assert m.case_count == 0
        assert m.zero_cost_verified is True
        assert m.zero_tokens_verified is True

    def test_phase3_comparison_metrics_new_fields_roundtrip(self):
        """All new Phase3ComparisonMetrics fields survive a JSON roundtrip."""
        from aml_copilot.schemas import Phase3ComparisonMetrics, Phase3FrameworkMetrics
        fw = Phase3FrameworkMetrics(
            framework="test",
            disposition_accuracy=1.0,
            false_clear_rate_weighted=0.0,
            override_rate=0.0,
            human_review_rate=0.0,
            latency_p50_ms=1.0,
            latency_p95_ms=2.0,
            loc=100,
            total_cost_usd=0.0,
            eval_size=90,
        )
        from datetime import datetime, timezone
        cm = Phase3ComparisonMetrics(
            generated_at=datetime.now(tz=timezone.utc),
            eval_size=90,
            protocol_version="1.0",
            framework_version_information={"python": "3.11.7"},
            phase1_accuracy=0.7556,
            phase2_accuracy=0.7889,
            frameworks=[fw],
            all_dispositions_agree=True,
            all_reasoning_agree=True,
            all_human_review_flags_agree=True,
            all_costs_zero=True,
            all_tokens_zero=True,
            comparison_passed=True,
        )
        cm2 = Phase3ComparisonMetrics.model_validate_json(cm.model_dump_json())
        assert cm2.phase1_accuracy == pytest.approx(0.7556)
        assert cm2.phase2_accuracy == pytest.approx(0.7889)
        assert cm2.all_reasoning_agree is True
        assert cm2.all_human_review_flags_agree is True
        assert cm2.all_costs_zero is True
        assert cm2.all_tokens_zero is True
        assert cm2.comparison_passed is True
        assert cm2.framework_version_information == {"python": "3.11.7"}


# ══════════════════════════════════════════════════════════════════════════════
# 15. M5 Metrics computation (no fixture files)
# ══════════════════════════════════════════════════════════════════════════════

class TestM5MetricsComputation:
    """Unit tests for compute_framework_metrics() with synthetic data."""

    def _run_metrics(
        self,
        n_correct: int = 90,
        n_override: int = 5,
        n_human: int = 15,
        latencies: list[float] | None = None,
        cost_per_case: float = 0.0,
        tokens_per_case: int = 0,
        tmp_path=None,
    ):
        from aml_copilot.phase3_compare.metrics import compute_framework_metrics
        from aml_copilot.phase3_compare._shared import EXPECTED_EVAL_SIZE

        cases = [_m5_eval_case(f"CASE-{i:03d}", gold_label="ESCALATE") for i in range(EXPECTED_EVAL_SIZE)]
        results = []
        for i, case in enumerate(cases):
            disposition = "ESCALATE" if i < n_correct else "CLEAR"
            results.append(_m5_result(
                case.case_id,
                framework="langgraph",
                disposition=disposition,
                baseline_disposition="ESCALATE",
                agent_override=(i < n_override),
                human_review_flagged=(i < n_human),
                latency_ms=(latencies[i] if latencies else float(i + 1)),
                cost_usd=cost_per_case,
                tokens_used=tokens_per_case,
            ))

        runner_file = Path(__file__).parent.parent / "src/aml_copilot/phase3_compare/langgraph_runner.py"
        return compute_framework_metrics(results, cases, runner_file)

    def test_accuracy_computed_correctly(self):
        """Accuracy = correct / total."""
        m = self._run_metrics(n_correct=81)
        assert m.disposition_accuracy == pytest.approx(81 / 90)

    def test_override_rate_computed_correctly(self):
        """Override rate = overrides / total."""
        m = self._run_metrics(n_override=9)
        assert m.override_rate == pytest.approx(9 / 90)

    def test_human_review_rate_computed_correctly(self):
        """Human review rate = flagged / total."""
        m = self._run_metrics(n_human=15)
        assert m.human_review_rate == pytest.approx(15 / 90)

    def test_case_count_equals_input_length(self):
        """case_count equals the number of results passed."""
        m = self._run_metrics()
        assert m.case_count == 90

    def test_latency_stats_correct(self):
        """Latency p50, p95, avg, min, max are computed correctly."""
        import numpy as np
        lats = [float(i) for i in range(1, 91)]
        m = self._run_metrics(latencies=lats)
        assert m.latency_p50_ms == pytest.approx(np.percentile(lats, 50))
        assert m.latency_p95_ms == pytest.approx(np.percentile(lats, 95))
        assert m.average_latency_ms == pytest.approx(sum(lats) / 90)
        assert m.minimum_latency_ms == pytest.approx(1.0)
        assert m.maximum_latency_ms == pytest.approx(90.0)

    def test_zero_cost_verified_true_when_all_zero(self):
        """zero_cost_verified is True when all costs are 0.0."""
        m = self._run_metrics(cost_per_case=0.0)
        assert m.zero_cost_verified is True

    def test_zero_cost_verified_false_when_nonzero(self):
        """zero_cost_verified is False when any cost is non-zero."""
        m = self._run_metrics(cost_per_case=0.001)
        assert m.zero_cost_verified is False

    def test_zero_tokens_verified_true_when_all_zero(self):
        """zero_tokens_verified is True when all tokens are 0."""
        m = self._run_metrics(tokens_per_case=0)
        assert m.zero_tokens_verified is True

    def test_zero_tokens_verified_false_when_nonzero(self):
        """zero_tokens_verified is False when any tokens are non-zero."""
        m = self._run_metrics(tokens_per_case=5)
        assert m.zero_tokens_verified is False

    def test_loc_is_positive(self):
        """LOC for a real runner file is positive."""
        m = self._run_metrics()
        assert m.loc > 0

    def test_eval_size_matches_input(self):
        """eval_size equals the number of eval cases."""
        m = self._run_metrics()
        assert m.eval_size == 90

    def test_empty_results_raises(self):
        """compute_framework_metrics raises ValueError on empty results."""
        from aml_copilot.phase3_compare.metrics import compute_framework_metrics
        runner_file = Path(__file__).parent
        with pytest.raises(ValueError, match="empty"):
            compute_framework_metrics([], _m5_eval_cases_90(), runner_file)


# ══════════════════════════════════════════════════════════════════════════════
# 16. M5 Agreement checks (no fixture files)
# ══════════════════════════════════════════════════════════════════════════════

class TestM5AgreementChecks:
    """Unit tests for check_framework_agreement() with synthetic data."""

    def _two_identical_frameworks(self, n: int = 5):
        fw1 = [_m5_result(f"CASE-{i:03d}", "langgraph") for i in range(n)]
        fw2 = [_m5_result(f"CASE-{i:03d}", "crewai") for i in range(n)]
        return {"langgraph": fw1, "crewai": fw2}

    def test_all_agree_when_identical(self):
        """All agreement flags True when two frameworks produce identical results."""
        from aml_copilot.phase3_compare.metrics import check_framework_agreement
        ag = check_framework_agreement(self._two_identical_frameworks())
        assert ag["all_dispositions_agree"] is True
        assert ag["all_reasoning_agree"] is True
        assert ag["all_human_review_flags_agree"] is True
        assert ag["all_overrides_agree"] is True
        assert ag["all_costs_zero"] is True
        assert ag["all_tokens_zero"] is True

    def test_disposition_disagree_detected(self):
        """Differing disposition on one case triggers all_dispositions_agree=False."""
        from aml_copilot.phase3_compare.metrics import check_framework_agreement
        fw1 = [_m5_result("CASE-000", "langgraph", disposition="CLEAR")]
        fw2 = [_m5_result("CASE-000", "crewai",    disposition="ESCALATE")]
        ag = check_framework_agreement({"langgraph": fw1, "crewai": fw2})
        assert ag["all_dispositions_agree"] is False
        assert len(ag["disposition_disagreements"]) == 1
        assert ag["disposition_disagreements"][0][0] == "CASE-000"

    def test_decision_reason_disagree_detected(self):
        """Differing decision_reason triggers all_reasoning_agree=False."""
        from aml_copilot.phase3_compare.metrics import check_framework_agreement
        fw1 = [_m5_result("CASE-000", "langgraph", decision_reason="clear")]
        fw2 = [_m5_result("CASE-000", "crewai",    decision_reason="different")]
        ag = check_framework_agreement({"langgraph": fw1, "crewai": fw2})
        assert ag["all_reasoning_agree"] is False

    def test_agent_reasoning_disagree_detected(self):
        """Differing agent_reasoning triggers all_reasoning_agree=False."""
        from aml_copilot.phase3_compare.metrics import check_framework_agreement
        fw1 = [_m5_result("CASE-000", "langgraph", agent_reasoning="reasoning A")]
        fw2 = [_m5_result("CASE-000", "crewai",    agent_reasoning="reasoning B")]
        ag = check_framework_agreement({"langgraph": fw1, "crewai": fw2})
        assert ag["all_reasoning_agree"] is False

    def test_human_review_disagree_detected(self):
        """Differing human_review_flagged triggers all_human_review_flags_agree=False."""
        from aml_copilot.phase3_compare.metrics import check_framework_agreement
        fw1 = [_m5_result("CASE-000", "langgraph", human_review_flagged=True)]
        fw2 = [_m5_result("CASE-000", "crewai",    human_review_flagged=False)]
        ag = check_framework_agreement({"langgraph": fw1, "crewai": fw2})
        assert ag["all_human_review_flags_agree"] is False

    def test_nonzero_cost_detected(self):
        """Non-zero cost_usd triggers all_costs_zero=False."""
        from aml_copilot.phase3_compare.metrics import check_framework_agreement
        fw1 = [_m5_result("CASE-000", "langgraph", cost_usd=0.001)]
        fw2 = [_m5_result("CASE-000", "crewai",    cost_usd=0.0)]
        ag = check_framework_agreement({"langgraph": fw1, "crewai": fw2})
        assert ag["all_costs_zero"] is False
        assert len(ag["cost_errors"]) == 1

    def test_nonzero_tokens_detected(self):
        """Non-zero tokens_used triggers all_tokens_zero=False."""
        from aml_copilot.phase3_compare.metrics import check_framework_agreement
        fw1 = [_m5_result("CASE-000", "langgraph", tokens_used=5)]
        fw2 = [_m5_result("CASE-000", "crewai",    tokens_used=0)]
        ag = check_framework_agreement({"langgraph": fw1, "crewai": fw2})
        assert ag["all_tokens_zero"] is False
        assert len(ag["token_errors"]) == 1

    def test_empty_input_returns_all_true(self):
        """Empty framework_results dict → all agreement flags True."""
        from aml_copilot.phase3_compare.metrics import check_framework_agreement
        ag = check_framework_agreement({})
        assert ag["all_dispositions_agree"] is True
        assert ag["all_reasoning_agree"] is True

    def test_single_framework_returns_all_true(self):
        """Single framework (nothing to compare) → all agreement flags True."""
        from aml_copilot.phase3_compare.metrics import check_framework_agreement
        fw1 = [_m5_result(f"CASE-{i:03d}", "langgraph") for i in range(5)]
        ag = check_framework_agreement({"langgraph": fw1})
        assert ag["all_dispositions_agree"] is True

    def test_comparison_passed_false_on_disagreement(self):
        """compute_comparison_metrics sets comparison_passed=False on disagreement."""
        from aml_copilot.phase3_compare.metrics import compute_comparison_metrics
        fw1 = [_m5_result("CASE-000", "langgraph", disposition="CLEAR")]
        fw2 = [_m5_result("CASE-000", "crewai",    disposition="ESCALATE")]
        cases = [_m5_eval_case("CASE-000")]
        from aml_copilot.schemas import Phase3FrameworkMetrics
        fm = Phase3FrameworkMetrics(
            framework="langgraph",
            disposition_accuracy=0.5,
            false_clear_rate_weighted=0.0,
            override_rate=0.0,
            human_review_rate=0.0,
            latency_p50_ms=1.0,
            latency_p95_ms=1.0,
            loc=100,
            total_cost_usd=0.0,
            eval_size=1,
        )
        cm = compute_comparison_metrics(
            framework_results={"langgraph": fw1, "crewai": fw2},
            framework_metrics=[fm],
            eval_cases=cases,
            phase1_accuracy=0.75,
            phase2_accuracy=0.79,
        )
        assert cm.comparison_passed is False
        assert cm.all_dispositions_agree is False

    def test_comparison_passed_false_on_runner_error(self):
        """compute_comparison_metrics sets comparison_passed=False when a runner failed."""
        from aml_copilot.phase3_compare.metrics import compute_comparison_metrics
        fw1 = [_m5_result("CASE-000", "langgraph", disposition="CLEAR")]
        cases = [_m5_eval_case("CASE-000")]
        from aml_copilot.schemas import Phase3FrameworkMetrics
        fm = Phase3FrameworkMetrics(
            framework="langgraph",
            disposition_accuracy=1.0,
            false_clear_rate_weighted=0.0,
            override_rate=0.0,
            human_review_rate=0.0,
            latency_p50_ms=1.0,
            latency_p95_ms=1.0,
            loc=100,
            total_cost_usd=0.0,
            eval_size=1,
        )
        cm = compute_comparison_metrics(
            framework_results={"langgraph": fw1},
            framework_metrics=[fm],
            eval_cases=cases,
            phase1_accuracy=0.75,
            phase2_accuracy=0.79,
            runner_errors={"crewai": RuntimeError("Simulated failure")},
        )
        assert cm.comparison_passed is False


# ══════════════════════════════════════════════════════════════════════════════
# 17. M5 Runner isolation (no fixture files needed beyond monkeypatch)
# ══════════════════════════════════════════════════════════════════════════════

class TestM5RunnerIsolation:
    """Verify that runner failures are isolated and don't corrupt other results."""
    pytestmark = [pytest.mark.integration, pytest.mark.compare, pytest.mark.slow]

    def test_one_runner_failure_does_not_abort_others(self, monkeypatch, tmp_path, _require_fixture_files):
        """When CrewAI raises, LangGraph and OpenAI Agents still complete."""
        import aml_copilot.phase3_compare.run_comparison as comp_module
        from aml_copilot.phase3_compare.crewai_runner import CrewAIRunner

        def fail_run(self_inner, eval_path, baseline_path):
            raise RuntimeError("Simulated CrewAI failure")

        monkeypatch.setattr(CrewAIRunner, "run", fail_run)

        out = tmp_path / "comparison.json"
        comparison = comp_module.run(_EVAL_PATH, _BASELINE_PATH, out)

        # LangGraph and OpenAI Agents should still produce results
        fw_names = {m.framework for m in comparison.frameworks}
        assert "langgraph" in fw_names
        assert "openai_agents" in fw_names
        # CrewAI is absent due to failure
        assert "crewai" not in fw_names
        # comparison_passed must be False (runner failed)
        assert comparison.comparison_passed is False

    def test_all_runner_failures_returns_empty_frameworks(self, monkeypatch, tmp_path, _require_fixture_files):
        """When all runners fail, frameworks list is empty and comparison_passed=False."""
        from aml_copilot.phase3_compare.langgraph_runner import LangGraphRunner
        from aml_copilot.phase3_compare.crewai_runner import CrewAIRunner
        from aml_copilot.phase3_compare.openai_agents_runner import OpenAIAgentsRunner
        import aml_copilot.phase3_compare.run_comparison as comp_module

        def fail(self_inner, ep, bp):
            raise RuntimeError("All fail")

        monkeypatch.setattr(LangGraphRunner, "run", fail)
        monkeypatch.setattr(CrewAIRunner, "run", fail)
        monkeypatch.setattr(OpenAIAgentsRunner, "run", fail)

        out = tmp_path / "comparison.json"
        comparison = comp_module.run(_EVAL_PATH, _BASELINE_PATH, out)
        assert comparison.frameworks == []
        assert comparison.comparison_passed is False


# ══════════════════════════════════════════════════════════════════════════════
# 18. M5 Integration tests — require fixture files
# ══════════════════════════════════════════════════════════════════════════════

import tempfile as _tempfile


@pytest.fixture(scope="module")
def _comparison_result(_fixture_files_available):
    """Run the full comparison once; return (Phase3ComparisonMetrics, out_path)."""
    if not _fixture_files_available:
        pytest.skip("Fixture files not available")
    from aml_copilot.phase3_compare.run_comparison import run
    tmpdir = _tempfile.mkdtemp()
    out = Path(tmpdir) / "phase3_comparison_metrics.json"
    comparison = run(_EVAL_PATH, _BASELINE_PATH, out)
    return comparison, out


@pytest.fixture(scope="module")
def _comparison_metrics(_comparison_result):
    return _comparison_result[0]


@pytest.fixture(scope="module")
def _comparison_out_path(_comparison_result):
    return _comparison_result[1]


class TestM5ComparisonIntegration:
    """Integration tests for run_comparison.run() against real fixture files."""
    pytestmark = [pytest.mark.integration, pytest.mark.compare, pytest.mark.slow]

    # ── Core acceptance criteria ──────────────────────────────────────────────

    def test_comparison_passed(self, _comparison_metrics):
        """comparison_passed is True — all agreement checks pass."""
        assert _comparison_metrics.comparison_passed is True

    def test_all_dispositions_agree(self, _comparison_metrics):
        """all_dispositions_agree is True — 90/90 match across frameworks."""
        assert _comparison_metrics.all_dispositions_agree is True

    def test_all_reasoning_agree(self, _comparison_metrics):
        """all_reasoning_agree is True — 90/90 match across frameworks."""
        assert _comparison_metrics.all_reasoning_agree is True

    def test_all_human_review_flags_agree(self, _comparison_metrics):
        """all_human_review_flags_agree is True — 90/90 match."""
        assert _comparison_metrics.all_human_review_flags_agree is True

    def test_all_costs_zero(self, _comparison_metrics):
        """all_costs_zero is True — zero API spend across all frameworks."""
        assert _comparison_metrics.all_costs_zero is True

    def test_all_tokens_zero(self, _comparison_metrics):
        """all_tokens_zero is True — zero token consumption."""
        assert _comparison_metrics.all_tokens_zero is True

    def test_accuracy_matches_expected(self, _comparison_metrics):
        """All framework accuracies equal the committed Phase 2 metric."""
        for m in _comparison_metrics.frameworks:
            assert m.disposition_accuracy == pytest.approx(
                _EXPECTED_P2_ACCURACY, abs=_ACCURACY_TOL
            ), f"{m.framework} accuracy {m.disposition_accuracy} != {_EXPECTED_P2_ACCURACY}"

    def test_eval_size_90(self, _comparison_metrics):
        """eval_size is 90."""
        assert _comparison_metrics.eval_size == 90

    def test_three_framework_entries(self, _comparison_metrics):
        """Comparison contains exactly 3 framework entries."""
        assert len(_comparison_metrics.frameworks) == 3

    def test_framework_order_matches_registry(self, _comparison_metrics):
        """Framework entries appear in registry order: langgraph, crewai, openai_agents."""
        from aml_copilot.phase3_compare.run_comparison import RUNNER_REGISTRY
        registry_order = [cls.framework_name for cls, _ in RUNNER_REGISTRY]
        result_order = [m.framework for m in _comparison_metrics.frameworks]
        assert result_order == registry_order

    def test_zero_cost_per_framework(self, _comparison_metrics):
        """Every framework's zero_cost_verified flag is True."""
        for m in _comparison_metrics.frameworks:
            assert m.zero_cost_verified is True, f"{m.framework} has non-zero cost"

    def test_zero_tokens_per_framework(self, _comparison_metrics):
        """Every framework's zero_tokens_verified flag is True."""
        for m in _comparison_metrics.frameworks:
            assert m.zero_tokens_verified is True, f"{m.framework} has non-zero tokens"

    def test_case_count_90_per_framework(self, _comparison_metrics):
        """Every framework processed exactly 90 cases."""
        for m in _comparison_metrics.frameworks:
            assert m.case_count == 90, f"{m.framework} case_count={m.case_count}"

    def test_loc_positive_per_framework(self, _comparison_metrics):
        """Every framework has LOC > 0."""
        for m in _comparison_metrics.frameworks:
            assert m.loc > 0, f"{m.framework} has LOC=0"

    def test_latency_positive_per_framework(self, _comparison_metrics):
        """Every framework has positive p50, p95, average, min, max latency."""
        for m in _comparison_metrics.frameworks:
            assert m.latency_p50_ms > 0, f"{m.framework} p50=0"
            assert m.latency_p95_ms > 0, f"{m.framework} p95=0"
            assert m.average_latency_ms > 0, f"{m.framework} avg=0"
            assert m.minimum_latency_ms > 0, f"{m.framework} min=0"

    def test_protocol_version_is_1_0(self, _comparison_metrics):
        """protocol_version is '1.0'."""
        assert _comparison_metrics.protocol_version == "1.0"

    def test_phase1_accuracy_loaded(self, _comparison_metrics):
        """phase1_accuracy is the known Phase 1 baseline value."""
        assert _comparison_metrics.phase1_accuracy == pytest.approx(0.7556, abs=0.001)

    def test_phase2_accuracy_loaded(self, _comparison_metrics):
        """phase2_accuracy is the known Phase 2 LangGraph value."""
        assert _comparison_metrics.phase2_accuracy == pytest.approx(
            _EXPECTED_P2_ACCURACY, abs=_ACCURACY_TOL
        )

    def test_framework_version_info_present(self, _comparison_metrics):
        """framework_version_information contains at least 'python'."""
        assert "python" in _comparison_metrics.framework_version_information

    # ── Artifact ──────────────────────────────────────────────────────────────

    def test_artifact_file_written(self, _comparison_out_path):
        """Output artifact file exists on disk after comparison run."""
        assert _comparison_out_path.exists()

    def test_artifact_is_valid_json(self, _comparison_out_path):
        """Output artifact is parseable as JSON."""
        import json
        data = json.loads(_comparison_out_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_artifact_schema_valid(self, _comparison_out_path):
        """Output artifact parses as Phase3ComparisonMetrics via Pydantic."""
        from aml_copilot.schemas import Phase3ComparisonMetrics
        cm = Phase3ComparisonMetrics.model_validate_json(
            _comparison_out_path.read_text(encoding="utf-8")
        )
        assert cm.comparison_passed is True
        assert cm.eval_size == 90

    def test_artifact_roundtrips_identically(self, _comparison_out_path):
        """Re-parsing and re-serialising the artifact produces identical JSON."""
        from aml_copilot.schemas import Phase3ComparisonMetrics
        raw = _comparison_out_path.read_text(encoding="utf-8")
        cm = Phase3ComparisonMetrics.model_validate_json(raw)
        assert Phase3ComparisonMetrics.model_validate_json(
            cm.model_dump_json()
        ).comparison_passed is True

    # ── CLI ───────────────────────────────────────────────────────────────────

    def test_cli_exits_0(self, tmp_path, _require_fixture_files):
        """run_comparison CLI exits with code 0 when all frameworks agree."""
        import subprocess
        out = tmp_path / "cli_comparison.json"
        result = subprocess.run(
            [
                sys.executable, "-m",
                "aml_copilot.phase3_compare.run_comparison",
                "--eval", str(_EVAL_PATH),
                "--baseline", str(_BASELINE_PATH),
                "--out", str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_cli_output_contains_pass(self, tmp_path, _require_fixture_files):
        """CLI stdout contains 'PASS' verdict."""
        import subprocess
        out = tmp_path / "cli_comparison2.json"
        result = subprocess.run(
            [
                sys.executable, "-m",
                "aml_copilot.phase3_compare.run_comparison",
                "--eval", str(_EVAL_PATH),
                "--baseline", str(_BASELINE_PATH),
                "--out", str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert "PASS" in result.stdout, (
            f"'PASS' not found in CLI output:\n{result.stdout}"
        )

    def test_cli_output_contains_table_header(self, tmp_path, _require_fixture_files):
        """CLI stdout contains the comparison table header."""
        import subprocess
        out = tmp_path / "cli_comparison3.json"
        result = subprocess.run(
            [
                sys.executable, "-m",
                "aml_copilot.phase3_compare.run_comparison",
                "--eval", str(_EVAL_PATH),
                "--baseline", str(_BASELINE_PATH),
                "--out", str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert "Phase 3 Framework Comparison" in result.stdout
        assert "Framework Metrics" in result.stdout
        assert "Agreement Summary" in result.stdout

    def test_cli_creates_artifact(self, tmp_path, _require_fixture_files):
        """CLI creates the artifact file at the --out path."""
        import subprocess
        out = tmp_path / "cli_artifact.json"
        subprocess.run(
            [
                sys.executable, "-m",
                "aml_copilot.phase3_compare.run_comparison",
                "--eval", str(_EVAL_PATH),
                "--baseline", str(_BASELINE_PATH),
                "--out", str(out),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.exists()

    # ── Deterministic ordering ────────────────────────────────────────────────

    def test_deterministic_on_repeated_runs(self, _require_fixture_files):
        """Two consecutive runs produce identical framework order and metrics."""
        from aml_copilot.phase3_compare.run_comparison import run
        out1 = Path(_tempfile.mktemp(suffix=".json"))
        out2 = Path(_tempfile.mktemp(suffix=".json"))
        try:
            c1 = run(_EVAL_PATH, _BASELINE_PATH, out1)
            c2 = run(_EVAL_PATH, _BASELINE_PATH, out2)
            order1 = [m.framework for m in c1.frameworks]
            order2 = [m.framework for m in c2.frameworks]
            assert order1 == order2
            for m1, m2 in zip(c1.frameworks, c2.frameworks):
                assert m1.framework == m2.framework
                assert m1.disposition_accuracy == pytest.approx(m2.disposition_accuracy)
                assert m1.false_clear_rate_weighted == pytest.approx(m2.false_clear_rate_weighted)
        finally:
            out1.unlink(missing_ok=True)
            out2.unlink(missing_ok=True)

    # ── Disagreement injection ────────────────────────────────────────────────

    def test_injected_disagreement_sets_comparison_passed_false(self, monkeypatch, tmp_path, _require_fixture_files):
        """Injecting a wrong disposition into one framework → comparison_passed=False."""
        from aml_copilot.phase3_compare.langgraph_runner import LangGraphRunner
        import aml_copilot.phase3_compare.run_comparison as comp_module

        original_run = LangGraphRunner.run

        def patched_run(self_inner, eval_path, baseline_path):
            results = original_run(self_inner, eval_path, baseline_path)
            # Flip the first case's disposition
            first = results[0]
            wrong_disp = "CLEAR" if first.disposition == "ESCALATE" else "ESCALATE"
            results[0] = first.model_copy(update={"disposition": wrong_disp})
            return results

        monkeypatch.setattr(LangGraphRunner, "run", patched_run)

        out = tmp_path / "injected.json"
        comparison = comp_module.run(_EVAL_PATH, _BASELINE_PATH, out)
        assert comparison.comparison_passed is False
        assert comparison.all_dispositions_agree is False


# ══════════════════════════════════════════════════════════════════════════════
# 19. M6 Mini-fixture tests — fully offline, no generated artifacts required
# ══════════════════════════════════════════════════════════════════════════════

_MINI_EVAL = Path(__file__).parent / "fixtures" / "phase3_mini_eval.jsonl"
_MINI_BASELINE = Path(__file__).parent / "fixtures" / "phase3_mini_baseline.jsonl"
_MINI_EXPECTED = Path(__file__).parent / "fixtures" / "phase3_expected_comparison.json"
_MINI_CASE_IDS = ["MINI_SH_001", "MINI_R3_001", "MINI_AN_001", "MINI_EL_001", "MINI_CL_001"]
_MINI_N = 5


@pytest.fixture(scope="module")
def _mini_expected() -> dict:
    """Load the committed expected-comparison fixture."""
    import json
    return json.loads(_MINI_EXPECTED.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def _mini_langgraph_results() -> "list[Phase3CaseResult]":
    from aml_copilot.phase3_compare.testing import run_mini_langgraph
    return run_mini_langgraph()


@pytest.fixture(scope="module")
def _mini_crewai_results() -> "list[Phase3CaseResult]":
    from aml_copilot.phase3_compare.testing import run_mini_crewai
    return run_mini_crewai()


@pytest.fixture(scope="module")
def _mini_openai_agents_results() -> "list[Phase3CaseResult]":
    from aml_copilot.phase3_compare.testing import run_mini_openai_agents
    return run_mini_openai_agents()


@pytest.fixture(scope="module")
def _mini_all_results(
    _mini_langgraph_results,
    _mini_crewai_results,
    _mini_openai_agents_results,
) -> "dict[str, list[Phase3CaseResult]]":
    return {
        "langgraph":      _mini_langgraph_results,
        "crewai":         _mini_crewai_results,
        "openai_agents":  _mini_openai_agents_results,
    }


@pytest.fixture(scope="module")
def _mini_comparison_result(_mini_all_results, _mini_expected) -> "Phase3ComparisonMetrics":
    """Run the full comparison pipeline against mini fixtures."""
    import tempfile
    from aml_copilot.phase3_compare.run_comparison import run
    tmp_out = Path(tempfile.mktemp(suffix=".json"))
    cm = run(_MINI_EVAL, _MINI_BASELINE, tmp_out)
    tmp_out.unlink(missing_ok=True)
    return cm


class TestM6MiniFixtures:
    """Offline mini-fixture tests — no generated artifacts, no API keys, no network."""

    pytestmark = [pytest.mark.compare]

    # ── 1. Schema validity ────────────────────────────────────────────────────

    def test_mini_eval_schema_valid(self):
        """All 5 mini eval cases parse as EvalCase without error."""
        from aml_copilot.schemas import EvalCase
        cases = []
        with open(_MINI_EVAL, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    cases.append(EvalCase.model_validate_json(line))
        assert len(cases) == _MINI_N

    def test_mini_baseline_schema_valid(self):
        """All 5 mini baseline rows parse as CaseResult without error."""
        from aml_copilot.schemas import CaseResult
        rows = []
        with open(_MINI_BASELINE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(CaseResult.model_validate_json(line))
        assert len(rows) == _MINI_N

    def test_mini_expected_json_valid(self, _mini_expected):
        """Expected comparison JSON contains all required keys."""
        for key in ("eval_size", "comparison_passed", "all_dispositions_agree",
                    "all_reasoning_agree", "all_human_review_flags_agree",
                    "all_costs_zero", "all_tokens_zero", "cases", "per_framework"):
            assert key in _mini_expected, f"Missing key: {key}"

    # ── 2. Branch coverage ────────────────────────────────────────────────────

    def test_five_branches_represented(self):
        """Exactly 5 unique branch cases are present (one per mock_llm branch)."""
        from aml_copilot.schemas import EvalCase
        cases = []
        with open(_MINI_EVAL, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    cases.append(EvalCase.model_validate_json(line))
        assert len(cases) == _MINI_N
        case_ids = [c.case_id for c in cases]
        assert set(case_ids) == set(_MINI_CASE_IDS)

    def test_sanctions_branch_covered(self):
        """MINI_SH_001 provides Branch 1 coverage (sanctions score >= 0.90)."""
        from aml_copilot.schemas import CaseResult
        rows = {}
        with open(_MINI_BASELINE, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = CaseResult.model_validate_json(line.strip())
                    rows[r.case_id] = r
        sh = rows["MINI_SH_001"]
        assert sh.sanctions_hits, "MINI_SH_001 must have at least one sanctions hit"
        assert sh.sanctions_hits[0].match_score >= 0.90

    def test_critical_rule_branch_covered(self):
        """MINI_R3_001 provides Branch 2 coverage (severity-3 rule)."""
        from aml_copilot.schemas import CaseResult
        rows = {}
        with open(_MINI_BASELINE, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = CaseResult.model_validate_json(line.strip())
                    rows[r.case_id] = r
        r3 = rows["MINI_R3_001"]
        assert any(f.severity == 3 for f in r3.rule_firings), "MINI_R3_001 must have severity-3 rule"

    def test_agent_extension_branch_covered(self):
        """MINI_AN_001 provides Branch 3 coverage (anomaly >= 0.90 + sev-2 rule)."""
        from aml_copilot.schemas import CaseResult
        rows = {}
        with open(_MINI_BASELINE, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = CaseResult.model_validate_json(line.strip())
                    rows[r.case_id] = r
        an = rows["MINI_AN_001"]
        assert an.anomaly_score is not None
        assert an.anomaly_score.percentile >= 0.90
        assert any(f.severity >= 2 for f in an.rule_firings)

    # ── 3. Framework mini runs ────────────────────────────────────────────────

    def test_langgraph_mini_returns_five(self, _mini_langgraph_results):
        """LangGraph produces exactly 5 results on mini fixture."""
        assert len(_mini_langgraph_results) == _MINI_N

    def test_crewai_mini_returns_five(self, _mini_crewai_results):
        """CrewAI produces exactly 5 results on mini fixture."""
        assert len(_mini_crewai_results) == _MINI_N

    def test_openai_agents_mini_returns_five(self, _mini_openai_agents_results):
        """OpenAI Agents produces exactly 5 results on mini fixture."""
        assert len(_mini_openai_agents_results) == _MINI_N

    # ── 4. Expected dispositions ──────────────────────────────────────────────

    def test_expected_dispositions_match(self, _mini_all_results, _mini_expected):
        """All frameworks produce dispositions matching the committed expected fixture."""
        expected_disp = {cid: v["disposition"] for cid, v in _mini_expected["cases"].items()}
        for fw, results in _mini_all_results.items():
            by_id = {r.case_id: r for r in results}
            for cid, exp in expected_disp.items():
                got = by_id[cid].disposition
                assert got == exp, f"{fw} [{cid}]: expected {exp!r}, got {got!r}"

    def test_expected_decision_reasons_match(self, _mini_all_results, _mini_expected):
        """All frameworks produce decision_reasons matching the committed expected fixture."""
        expected_reason = {cid: v["decision_reason"] for cid, v in _mini_expected["cases"].items()}
        for fw, results in _mini_all_results.items():
            by_id = {r.case_id: r for r in results}
            for cid, exp in expected_reason.items():
                got = by_id[cid].decision_reason
                assert got == exp, f"{fw} [{cid}]: reason expected {exp!r}, got {got!r}"

    def test_expected_human_review_flags_match(self, _mini_all_results, _mini_expected):
        """All frameworks produce human_review_flagged matching the committed expected fixture."""
        expected_hr = {cid: v["human_review_flagged"] for cid, v in _mini_expected["cases"].items()}
        for fw, results in _mini_all_results.items():
            by_id = {r.case_id: r for r in results}
            for cid, exp in expected_hr.items():
                got = by_id[cid].human_review_flagged
                assert got == exp, f"{fw} [{cid}]: human_review expected {exp!r}, got {got!r}"

    def test_expected_agent_override_match(self, _mini_all_results, _mini_expected):
        """agent_override for MINI_AN_001 is True (agent extension overrides baseline CLEAR)."""
        expected_ov = {cid: v["agent_override"] for cid, v in _mini_expected["cases"].items()}
        for fw, results in _mini_all_results.items():
            by_id = {r.case_id: r for r in results}
            for cid, exp in expected_ov.items():
                got = by_id[cid].agent_override
                assert got == exp, f"{fw} [{cid}]: agent_override expected {exp!r}, got {got!r}"

    # ── 5. Three-way parity ───────────────────────────────────────────────────

    def test_three_way_parity_dispositions(self, _mini_all_results):
        """LangGraph, CrewAI, and OpenAI Agents produce identical dispositions on mini fixture."""
        from aml_copilot.phase3_compare.metrics import check_framework_agreement
        ag = check_framework_agreement(_mini_all_results)
        assert ag["all_dispositions_agree"] is True, (
            f"Disposition disagreements: {ag['disposition_disagreements']}"
        )

    def test_three_way_parity_reasoning(self, _mini_all_results):
        """All three frameworks produce identical decision_reason and agent_reasoning."""
        from aml_copilot.phase3_compare.metrics import check_framework_agreement
        ag = check_framework_agreement(_mini_all_results)
        assert ag["all_reasoning_agree"] is True, (
            f"Reasoning disagreements: {ag['reasoning_disagreements']}"
        )

    def test_three_way_parity_human_review(self, _mini_all_results):
        """All three frameworks produce identical human_review_flagged values."""
        from aml_copilot.phase3_compare.metrics import check_framework_agreement
        ag = check_framework_agreement(_mini_all_results)
        assert ag["all_human_review_flags_agree"] is True, (
            f"Human review disagreements: {ag['human_review_disagreements']}"
        )

    # ── 6. Cost and token constraints ─────────────────────────────────────────

    def test_zero_tokens_all_frameworks(self, _mini_all_results):
        """tokens_used == 0 for every result in every framework."""
        for fw, results in _mini_all_results.items():
            bad = [r.case_id for r in results if r.tokens_used != 0]
            assert bad == [], f"{fw} has non-zero tokens: {bad}"

    def test_zero_cost_all_frameworks(self, _mini_all_results):
        """cost_usd == 0.0 for every result in every framework."""
        for fw, results in _mini_all_results.items():
            bad = [r.case_id for r in results if r.cost_usd != 0.0]
            assert bad == [], f"{fw} has non-zero cost: {bad}"

    # ── 7. Network and API key isolation ──────────────────────────────────────

    def test_no_network_calls_langgraph(self, monkeypatch):
        """LangGraph mini-run makes zero HTTP calls."""
        import httpx
        def fail_http(self_inner, *a, **kw):
            raise AssertionError("Network call detected in LangGraph mini-run")
        monkeypatch.setattr(httpx.Client, "send", fail_http)
        monkeypatch.setattr(httpx.AsyncClient, "send", fail_http)
        from aml_copilot.phase3_compare.testing import run_mini_langgraph
        results = run_mini_langgraph()
        assert len(results) == _MINI_N

    def test_no_network_calls_crewai(self, monkeypatch):
        """CrewAI mini-run makes zero HTTP calls."""
        import httpx
        def fail_http(self_inner, *a, **kw):
            raise AssertionError("Network call detected in CrewAI mini-run")
        monkeypatch.setattr(httpx.Client, "send", fail_http)
        monkeypatch.setattr(httpx.AsyncClient, "send", fail_http)
        from aml_copilot.phase3_compare.testing import run_mini_crewai
        results = run_mini_crewai()
        assert len(results) == _MINI_N

    def test_no_network_calls_openai_agents(self, monkeypatch):
        """OpenAI Agents mini-run makes zero HTTP calls."""
        import httpx
        def fail_http(self_inner, *a, **kw):
            raise AssertionError("Network call detected in OpenAI Agents mini-run")
        monkeypatch.setattr(httpx.Client, "send", fail_http)
        monkeypatch.setattr(httpx.AsyncClient, "send", fail_http)
        from aml_copilot.phase3_compare.testing import run_mini_openai_agents
        results = run_mini_openai_agents()
        assert len(results) == _MINI_N

    def test_no_api_key_required(self, monkeypatch):
        """All three runners work when OPENAI_API_KEY is absent."""
        for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID",
                    "OPENAI_PROJECT_ID", "OPENAI_AGENTS_DISABLE_TRACING"):
            monkeypatch.delenv(var, raising=False)
        from aml_copilot.phase3_compare.testing import (
            run_mini_langgraph, run_mini_crewai, run_mini_openai_agents,
        )
        assert len(run_mini_langgraph()) == _MINI_N
        assert len(run_mini_crewai()) == _MINI_N
        assert len(run_mini_openai_agents()) == _MINI_N

    def test_environment_variables_unchanged_after_run(self, monkeypatch):
        """Runner execution does not modify environment variables."""
        import os
        monkeypatch.setenv("OPENAI_API_KEY", "SENTINEL_KEY")
        from aml_copilot.phase3_compare.testing import run_mini_langgraph
        run_mini_langgraph()
        assert os.environ.get("OPENAI_API_KEY") == "SENTINEL_KEY"

    # ── 8. No raw data paths accessed ─────────────────────────────────────────

    def test_no_raw_data_path_accessed(self, monkeypatch):
        """Mini-fixture run does not open any path under data/raw/ or data/processed/."""
        import builtins
        _orig_open = builtins.open
        accessed_paths: list[str] = []

        def _tracking_open(file, *args, **kwargs):
            accessed_paths.append(str(file))
            return _orig_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", _tracking_open)
        from aml_copilot.phase3_compare.testing import run_mini_langgraph
        run_mini_langgraph()

        forbidden = [p for p in accessed_paths if "/data/raw/" in p or "/data/processed/" in p]
        assert forbidden == [], f"Opened raw/processed data paths: {forbidden}"

    # ── 9. Comparison CLI with mini fixtures ──────────────────────────────────

    def test_mini_comparison_cli_exits_0(self, tmp_path):
        """run_comparison CLI exits 0 with mini fixtures."""
        import subprocess
        out = tmp_path / "mini_cli_out.json"
        result = subprocess.run(
            [sys.executable, "-m", "aml_copilot.phase3_compare.run_comparison",
             "--eval", str(_MINI_EVAL),
             "--baseline", str(_MINI_BASELINE),
             "--out", str(out)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"CLI failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}"
        )

    def test_mini_comparison_cli_output_pass(self, tmp_path):
        """run_comparison CLI prints PASS with mini fixtures."""
        import subprocess
        out = tmp_path / "mini_cli_pass.json"
        result = subprocess.run(
            [sys.executable, "-m", "aml_copilot.phase3_compare.run_comparison",
             "--eval", str(_MINI_EVAL),
             "--baseline", str(_MINI_BASELINE),
             "--out", str(out)],
            capture_output=True, text=True,
        )
        assert "PASS" in result.stdout

    def test_mini_comparison_artifact_schema_valid(self, tmp_path):
        """Artifact produced from mini fixtures parses as Phase3ComparisonMetrics."""
        import subprocess
        from aml_copilot.schemas import Phase3ComparisonMetrics
        out = tmp_path / "mini_cli_schema.json"
        subprocess.run(
            [sys.executable, "-m", "aml_copilot.phase3_compare.run_comparison",
             "--eval", str(_MINI_EVAL),
             "--baseline", str(_MINI_BASELINE),
             "--out", str(out)],
            check=True, capture_output=True,
        )
        cm = Phase3ComparisonMetrics.model_validate_json(out.read_text(encoding="utf-8"))
        assert cm.comparison_passed is True
        assert cm.eval_size == _MINI_N

    # ── 10. comparison_passed and agreement flags ──────────────────────────────

    def test_mini_comparison_passed(self, _mini_comparison_result):
        """comparison_passed is True for mini fixtures."""
        assert _mini_comparison_result.comparison_passed is True

    def test_mini_all_dispositions_agree(self, _mini_comparison_result):
        """all_dispositions_agree is True for mini fixtures."""
        assert _mini_comparison_result.all_dispositions_agree is True

    def test_mini_all_reasoning_agree(self, _mini_comparison_result):
        """all_reasoning_agree is True for mini fixtures."""
        assert _mini_comparison_result.all_reasoning_agree is True

    def test_mini_all_human_review_agree(self, _mini_comparison_result):
        """all_human_review_flags_agree is True for mini fixtures."""
        assert _mini_comparison_result.all_human_review_flags_agree is True

    def test_mini_all_costs_zero(self, _mini_comparison_result):
        """all_costs_zero is True for mini fixtures."""
        assert _mini_comparison_result.all_costs_zero is True

    def test_mini_all_tokens_zero(self, _mini_comparison_result):
        """all_tokens_zero is True for mini fixtures."""
        assert _mini_comparison_result.all_tokens_zero is True

    def test_mini_eval_size(self, _mini_comparison_result):
        """eval_size equals 5 in mini comparison."""
        assert _mini_comparison_result.eval_size == _MINI_N

    def test_mini_three_frameworks(self, _mini_comparison_result):
        """Mini comparison includes results for all three frameworks."""
        assert len(_mini_comparison_result.frameworks) == 3

    # ── 11. Validation error cases (no fixtures needed) ───────────────────────

    def test_empty_result_set_fails_validation(self):
        """validate_phase3_results raises on empty results list."""
        from aml_copilot.phase3_compare._shared import validate_phase3_results
        from aml_copilot.schemas import EvalCase
        case = EvalCase(case_id="C1", account_id="A1", gold_label="CLEAR",
                        case_type="ibm_labeled", relevant_txn_ids=[], notes="")
        with pytest.raises(RuntimeError, match="0 results"):
            validate_phase3_results([], [case], "langgraph")

    def test_duplicate_case_ids_fail_validation(self):
        """validate_phase3_results raises on duplicate case_ids."""
        from aml_copilot.phase3_compare._shared import validate_phase3_results
        from aml_copilot.schemas import EvalCase, Phase3CaseResult

        def _r(cid: str) -> Phase3CaseResult:
            return Phase3CaseResult(
                framework="langgraph", case_id=cid, account_id="A",
                disposition="CLEAR", decision_reason="clear",
                sanctions_hits=[], rule_firings=[], anomaly_score=None,
                latency_ms=1.0, agent_reasoning="", agent_override=False,
                baseline_disposition="CLEAR", human_review_flagged=False,
            )
        results = [_r("CASE-001"), _r("CASE-001")]  # duplicate
        cases = [EvalCase(case_id="CASE-001", account_id="A", gold_label="CLEAR",
                          case_type="ibm_labeled", relevant_txn_ids=[], notes=""),
                 EvalCase(case_id="CASE-002", account_id="A", gold_label="CLEAR",
                          case_type="ibm_labeled", relevant_txn_ids=[], notes="")]
        with pytest.raises(RuntimeError, match="duplicate"):
            validate_phase3_results(results, cases, "langgraph")

    def test_framework_tag_mismatch_fails_validation(self):
        """validate_phase3_results raises when result carries wrong framework tag."""
        from aml_copilot.phase3_compare._shared import validate_phase3_results
        from aml_copilot.schemas import EvalCase, Phase3CaseResult

        result = Phase3CaseResult(
            framework="crewai", case_id="CASE-001", account_id="A",
            disposition="CLEAR", decision_reason="clear",
            sanctions_hits=[], rule_firings=[], anomaly_score=None,
            latency_ms=1.0, agent_reasoning="", agent_override=False,
            baseline_disposition="CLEAR", human_review_flagged=False,
        )
        case = EvalCase(case_id="CASE-001", account_id="A", gold_label="CLEAR",
                        case_type="ibm_labeled", relevant_txn_ids=[], notes="")
        with pytest.raises(RuntimeError, match="wrong framework tag"):
            validate_phase3_results([result], [case], "langgraph")

    def test_non_zero_tokens_fail_zero_token_verification(self, tmp_path):
        """Zero-token verification catches non-zero tokens_used."""
        from aml_copilot.phase3_compare.metrics import check_framework_agreement
        from aml_copilot.schemas import Phase3CaseResult
        r = Phase3CaseResult(
            framework="langgraph", case_id="C1", account_id="A",
            disposition="CLEAR", decision_reason="clear",
            sanctions_hits=[], rule_firings=[], anomaly_score=None,
            latency_ms=1.0, agent_reasoning="", agent_override=False,
            baseline_disposition="CLEAR", human_review_flagged=False,
            tokens_used=5,
        )
        ag = check_framework_agreement({"langgraph": [r]})
        assert ag["all_tokens_zero"] is False
        assert len(ag["token_errors"]) == 1

    def test_non_zero_cost_fails_zero_cost_verification(self, tmp_path):
        """Zero-cost verification catches non-zero cost_usd."""
        from aml_copilot.phase3_compare.metrics import check_framework_agreement
        from aml_copilot.schemas import Phase3CaseResult
        r = Phase3CaseResult(
            framework="langgraph", case_id="C1", account_id="A",
            disposition="CLEAR", decision_reason="clear",
            sanctions_hits=[], rule_firings=[], anomaly_score=None,
            latency_ms=1.0, agent_reasoning="", agent_override=False,
            baseline_disposition="CLEAR", human_review_flagged=False,
            cost_usd=0.01,
        )
        ag = check_framework_agreement({"langgraph": [r]})
        assert ag["all_costs_zero"] is False
        assert len(ag["cost_errors"]) == 1

    # ── 12. Registry integrity ────────────────────────────────────────────────

    def test_duplicate_framework_registration_rejected(self):
        """Adding a duplicate framework name to RUNNER_REGISTRY causes a naming collision."""
        from aml_copilot.phase3_compare.run_comparison import RUNNER_REGISTRY
        names = [cls.framework_name for cls, _ in RUNNER_REGISTRY]
        assert len(names) == len(set(names)), (
            f"Duplicate framework names in registry: {names}"
        )

    def test_missing_framework_surfaced_in_comparison(self, monkeypatch, tmp_path):
        """A runner that raises is surfaced in failed-frameworks count, not silently hidden."""
        from aml_copilot.phase3_compare.langgraph_runner import LangGraphRunner
        import aml_copilot.phase3_compare.run_comparison as comp_module
        monkeypatch.setattr(LangGraphRunner, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("forced")))
        out = tmp_path / "missing_fw.json"
        cm = comp_module.run(_MINI_EVAL, _MINI_BASELINE, out)
        fw_names = {m.framework for m in cm.frameworks}
        assert "langgraph" not in fw_names
        assert cm.comparison_passed is False

    # ── 13. Deterministic ordering ────────────────────────────────────────────

    def test_deterministic_framework_ordering_mini(self, tmp_path):
        """Repeated mini comparison runs produce frameworks in registry order."""
        from aml_copilot.phase3_compare.run_comparison import run, RUNNER_REGISTRY
        registry_order = [cls.framework_name for cls, _ in RUNNER_REGISTRY]
        out1 = tmp_path / "det1.json"
        out2 = tmp_path / "det2.json"
        c1 = run(_MINI_EVAL, _MINI_BASELINE, out1)
        c2 = run(_MINI_EVAL, _MINI_BASELINE, out2)
        assert [m.framework for m in c1.frameworks] == registry_order
        assert [m.framework for m in c2.frameworks] == registry_order

    def test_repeated_runs_produce_identical_metrics(self, tmp_path):
        """Two consecutive mini comparison runs produce identical accuracy, FCR, overrides."""
        from aml_copilot.phase3_compare.run_comparison import run
        out1 = tmp_path / "rep1.json"
        out2 = tmp_path / "rep2.json"
        c1 = run(_MINI_EVAL, _MINI_BASELINE, out1)
        c2 = run(_MINI_EVAL, _MINI_BASELINE, out2)
        for m1, m2 in zip(c1.frameworks, c2.frameworks):
            assert m1.framework == m2.framework
            assert m1.disposition_accuracy == pytest.approx(m2.disposition_accuracy)
            assert m1.false_clear_rate_weighted == pytest.approx(m2.false_clear_rate_weighted)
            assert m1.override_rate == pytest.approx(m2.override_rate)
            assert m1.human_review_rate == pytest.approx(m2.human_review_rate)


# ══════════════════════════════════════════════════════════════════════════════
# 20. M6 Coverage boosters — unit tests for run_comparison formatters and helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_mini_comparison_metrics() -> "Phase3ComparisonMetrics":
    """Return a minimal Phase3ComparisonMetrics built from mini-fixture constants."""
    from datetime import datetime, timezone
    from aml_copilot.schemas import Phase3ComparisonMetrics, Phase3FrameworkMetrics
    fw = Phase3FrameworkMetrics(
        framework="langgraph",
        disposition_accuracy=1.0,
        false_clear_rate_weighted=0.0,
        override_rate=0.2,
        human_review_rate=0.4,
        latency_p50_ms=1.0,
        latency_p95_ms=2.0,
        average_latency_ms=1.5,
        minimum_latency_ms=0.5,
        maximum_latency_ms=3.0,
        case_count=5,
        loc=70,
        total_cost_usd=0.0,
        eval_size=5,
    )
    return Phase3ComparisonMetrics(
        generated_at=datetime.now(tz=timezone.utc),
        eval_size=5,
        protocol_version="1.0",
        framework_version_information={"python": "3.11.7"},
        phase1_accuracy=0.7556,
        phase2_accuracy=0.7889,
        frameworks=[fw],
        all_dispositions_agree=True,
        all_reasoning_agree=True,
        all_human_review_flags_agree=True,
        all_costs_zero=True,
        all_tokens_zero=True,
        comparison_passed=True,
    )


class TestRunComparisonFormatters:
    """Coverage for run_comparison._format_table, _yn, print_comparison, _load_accuracy."""

    pytestmark = [pytest.mark.compare]

    def test_format_table_contains_framework_name(self):
        """_format_table output contains the framework name."""
        from aml_copilot.phase3_compare.run_comparison import _format_table
        cm = _make_mini_comparison_metrics()
        table = _format_table(cm)
        assert "langgraph" in table

    def test_format_table_contains_numeric_values(self):
        """_format_table output contains formatted numeric values."""
        from aml_copilot.phase3_compare.run_comparison import _format_table
        cm = _make_mini_comparison_metrics()
        table = _format_table(cm)
        assert "1.0000" in table  # accuracy
        assert "0.0000" in table  # FCR

    def test_format_table_empty_frameworks(self):
        """_format_table produces header-only output when no frameworks."""
        from aml_copilot.phase3_compare.run_comparison import _format_table
        from datetime import datetime, timezone
        from aml_copilot.schemas import Phase3ComparisonMetrics
        cm = Phase3ComparisonMetrics(
            generated_at=datetime.now(tz=timezone.utc),
            eval_size=0, protocol_version="1.0",
            phase1_accuracy=0.0, phase2_accuracy=0.0,
            frameworks=[],
            all_dispositions_agree=True, all_reasoning_agree=True,
            all_human_review_flags_agree=True, all_costs_zero=True,
            all_tokens_zero=True, comparison_passed=False,
        )
        table = _format_table(cm)
        assert "Framework" in table
        assert "Accuracy" in table

    def test_yn_true(self):
        """_yn(True) returns a string starting with 'YES'."""
        from aml_copilot.phase3_compare.run_comparison import _yn
        assert _yn(True).startswith("YES")

    def test_yn_false(self):
        """_yn(False) returns a string starting with 'NO'."""
        from aml_copilot.phase3_compare.run_comparison import _yn
        assert _yn(False).startswith("NO")

    def test_yn_with_extra(self):
        """_yn includes extra annotation text."""
        from aml_copilot.phase3_compare.run_comparison import _yn
        assert "(90/90)" in _yn(True, "(90/90)")

    def test_print_comparison_pass_verdict(self, capsys):
        """print_comparison outputs PASS for a passing comparison."""
        from aml_copilot.phase3_compare.run_comparison import print_comparison
        cm = _make_mini_comparison_metrics()
        print_comparison(cm, {}, Path("/tmp/test.json"), 1.23)
        captured = capsys.readouterr()
        assert "PASS" in captured.out

    def test_print_comparison_fail_verdict(self, capsys):
        """print_comparison outputs FAIL for a failing comparison."""
        from aml_copilot.phase3_compare.run_comparison import print_comparison
        from datetime import datetime, timezone
        from aml_copilot.schemas import Phase3ComparisonMetrics
        cm = Phase3ComparisonMetrics(
            generated_at=datetime.now(tz=timezone.utc),
            eval_size=0, protocol_version="1.0",
            phase1_accuracy=0.0, phase2_accuracy=0.0,
            frameworks=[],
            all_dispositions_agree=False, all_reasoning_agree=True,
            all_human_review_flags_agree=True, all_costs_zero=True,
            all_tokens_zero=True, comparison_passed=False,
        )
        print_comparison(cm, {"crewai": RuntimeError("fail")}, Path("/tmp/fail.json"), 0.5)
        captured = capsys.readouterr()
        assert "FAIL" in captured.out

    def test_print_comparison_single_framework_agreement_section(self, capsys):
        """print_comparison handles single-framework (no agreement check)."""
        from aml_copilot.phase3_compare.run_comparison import print_comparison
        cm = _make_mini_comparison_metrics()
        print_comparison(cm, {}, Path("/tmp/t.json"), 0.1)
        captured = capsys.readouterr()
        assert "insufficient frameworks" in captured.out

    def test_load_accuracy_missing_file(self, tmp_path):
        """_load_accuracy returns 0.0 when the metrics file does not exist."""
        from aml_copilot.phase3_compare.run_comparison import _load_accuracy
        result = _load_accuracy(tmp_path / "nonexistent.json", "disposition_accuracy")
        assert result == pytest.approx(0.0)

    def test_load_accuracy_bad_key(self, tmp_path):
        """_load_accuracy returns 0.0 when the key is absent from the JSON."""
        import json
        from aml_copilot.phase3_compare.run_comparison import _load_accuracy
        f = tmp_path / "metrics.json"
        f.write_text(json.dumps({"other_key": 0.9}), encoding="utf-8")
        result = _load_accuracy(f, "disposition_accuracy")
        assert result == pytest.approx(0.0)

    def test_load_accuracy_valid_file(self, tmp_path):
        """_load_accuracy reads the correct value from a valid metrics file."""
        import json
        from aml_copilot.phase3_compare.run_comparison import _load_accuracy
        f = tmp_path / "metrics.json"
        f.write_text(json.dumps({"disposition_accuracy": 0.7889}), encoding="utf-8")
        result = _load_accuracy(f, "disposition_accuracy")
        assert result == pytest.approx(0.7889)

    def test_run_raises_on_missing_eval(self, tmp_path):
        """run() raises FileNotFoundError when eval_path does not exist."""
        from aml_copilot.phase3_compare.run_comparison import run
        with pytest.raises(FileNotFoundError, match="Eval set"):
            run(tmp_path / "missing_eval.jsonl", _MINI_BASELINE, tmp_path / "out.json")

    def test_run_raises_on_missing_baseline(self, tmp_path):
        """run() raises FileNotFoundError when baseline_path does not exist."""
        from aml_copilot.phase3_compare.run_comparison import run
        with pytest.raises(FileNotFoundError, match="Baseline"):
            run(_MINI_EVAL, tmp_path / "missing_baseline.jsonl", tmp_path / "out.json")

    def test_run_isolated_success(self):
        """_run_one_isolated returns (fw, results, None) on success."""
        from aml_copilot.phase3_compare.run_comparison import _run_one_isolated, RUNNER_REGISTRY
        runner_cls, runner_file = RUNNER_REGISTRY[0]  # langgraph
        fw, results, err = _run_one_isolated(runner_cls, runner_file, _MINI_EVAL, _MINI_BASELINE)
        assert fw == "langgraph"
        assert results is not None
        assert len(results) == _MINI_N
        assert err is None

    def test_run_isolated_failure(self):
        """_run_one_isolated returns (fw, None, exc) when runner raises."""
        from aml_copilot.phase3_compare.run_comparison import _run_one_isolated
        from aml_copilot.phase3_compare.langgraph_runner import LangGraphRunner

        class FailingRunner(LangGraphRunner):
            def run(self, ep, bp):
                raise RuntimeError("forced failure")

        FailingRunner.framework_name = "langgraph"  # satisfy fw attr
        _, runner_file = [(r, f) for r, f in __import__(
            "aml_copilot.phase3_compare.run_comparison", fromlist=["RUNNER_REGISTRY"]
        ).RUNNER_REGISTRY if r.framework_name == "langgraph"][0]

        fw, results, err = _run_one_isolated(FailingRunner, runner_file, _MINI_EVAL, _MINI_BASELINE)
        assert results is None
        assert isinstance(err, RuntimeError)


class TestOpenAIAgentsValidateShim:
    """Coverage for the _validate backward-compat shim in openai_agents_runner."""

    pytestmark = [pytest.mark.compare]

    def test_validate_shim_wrong_count_raises(self):
        """_validate shim raises when result count != EXPECTED_EVAL_SIZE."""
        from aml_copilot.phase3_compare.openai_agents_runner import _validate
        from aml_copilot.schemas import Phase3CaseResult

        def _r(cid: str) -> Phase3CaseResult:
            return Phase3CaseResult(
                framework="openai_agents", case_id=cid, account_id="A",
                disposition="CLEAR", decision_reason="clear",
                sanctions_hits=[], rule_firings=[], anomaly_score=None,
                latency_ms=1.0, agent_reasoning="", agent_override=False,
                baseline_disposition="CLEAR", human_review_flagged=False,
            )

        with pytest.raises(RuntimeError):
            _validate([_r("C1"), _r("C2")])  # << 90

    def test_validate_shim_duplicate_raises(self):
        """_validate shim raises on duplicate case_ids."""
        from aml_copilot.phase3_compare._shared import EXPECTED_EVAL_SIZE
        from aml_copilot.phase3_compare.openai_agents_runner import _validate
        from aml_copilot.schemas import Phase3CaseResult

        def _r(cid: str) -> Phase3CaseResult:
            return Phase3CaseResult(
                framework="openai_agents", case_id=cid, account_id="A",
                disposition="CLEAR", decision_reason="clear",
                sanctions_hits=[], rule_firings=[], anomaly_score=None,
                latency_ms=1.0, agent_reasoning="", agent_override=False,
                baseline_disposition="CLEAR", human_review_flagged=False,
            )

        results = [_r(f"CASE-{i:03d}") for i in range(EXPECTED_EVAL_SIZE - 1)]
        results.append(_r("CASE-000"))  # duplicate
        with pytest.raises(RuntimeError, match="duplicate"):
            _validate(results)

    def test_validate_shim_wrong_framework_raises(self):
        """_validate shim raises when framework tag != openai_agents."""
        from aml_copilot.phase3_compare._shared import EXPECTED_EVAL_SIZE
        from aml_copilot.phase3_compare.openai_agents_runner import _validate
        from aml_copilot.schemas import Phase3CaseResult

        results = [
            Phase3CaseResult(
                framework="langgraph", case_id=f"CASE-{i:03d}", account_id="A",
                disposition="CLEAR", decision_reason="clear",
                sanctions_hits=[], rule_firings=[], anomaly_score=None,
                latency_ms=1.0, agent_reasoning="", agent_override=False,
                baseline_disposition="CLEAR", human_review_flagged=False,
            )
            for i in range(EXPECTED_EVAL_SIZE)
        ]
        with pytest.raises(RuntimeError, match="wrong framework"):
            _validate(results)

    def test_stream_response_raises_not_implemented(self):
        """DeterministicAMLModel.stream_response raises NotImplementedError when awaited."""
        import asyncio
        from aml_copilot.phase3_compare.openai_agents_runner import DeterministicAMLModel

        async def _test():
            model = DeterministicAMLModel()
            await model.stream_response()

        with pytest.raises(NotImplementedError):
            asyncio.run(_test())

    def test_validate_shim_bad_tokens_raises(self):
        """_validate shim raises when tokens_used != 0."""
        from aml_copilot.phase3_compare._shared import EXPECTED_EVAL_SIZE
        from aml_copilot.phase3_compare.openai_agents_runner import _validate
        from aml_copilot.schemas import Phase3CaseResult

        results = [
            Phase3CaseResult(
                framework="openai_agents", case_id=f"CASE-{i:03d}", account_id="A",
                disposition="CLEAR", decision_reason="clear",
                sanctions_hits=[], rule_firings=[], anomaly_score=None,
                latency_ms=1.0, agent_reasoning="", agent_override=False,
                baseline_disposition="CLEAR", human_review_flagged=False,
                tokens_used=5,
            )
            for i in range(EXPECTED_EVAL_SIZE)
        ]
        with pytest.raises(RuntimeError, match="tokens"):
            _validate(results)


class TestSharedHelpersMissingCoverage:
    """Cover the missing-case error paths in validate_phase3_results."""

    pytestmark = [pytest.mark.compare]

    def test_validate_missing_eval_case_raises(self):
        """validate_phase3_results raises when result case_id not in eval set."""
        from aml_copilot.phase3_compare._shared import validate_phase3_results
        from aml_copilot.schemas import EvalCase, Phase3CaseResult

        result = Phase3CaseResult(
            framework="langgraph", case_id="EXTRA-001", account_id="A",
            disposition="CLEAR", decision_reason="clear",
            sanctions_hits=[], rule_firings=[], anomaly_score=None,
            latency_ms=1.0, agent_reasoning="", agent_override=False,
            baseline_disposition="CLEAR", human_review_flagged=False,
        )
        case = EvalCase(case_id="CASE-001", account_id="A", gold_label="CLEAR",
                        case_type="ibm_labeled", relevant_txn_ids=[], notes="")
        with pytest.raises(RuntimeError):
            validate_phase3_results([result], [case], "langgraph")

    def test_validate_empty_case_id_raises(self):
        """validate_phase3_results raises when result case_id is empty."""
        from aml_copilot.phase3_compare._shared import validate_phase3_results
        from aml_copilot.schemas import EvalCase, Phase3CaseResult

        result = Phase3CaseResult(
            framework="langgraph", case_id="", account_id="A",
            disposition="CLEAR", decision_reason="clear",
            sanctions_hits=[], rule_firings=[], anomaly_score=None,
            latency_ms=1.0, agent_reasoning="", agent_override=False,
            baseline_disposition="CLEAR", human_review_flagged=False,
        )
        case = EvalCase(case_id="", account_id="A", gold_label="CLEAR",
                        case_type="ibm_labeled", relevant_txn_ids=[], notes="")
        with pytest.raises(RuntimeError, match="missing required"):
            validate_phase3_results([result], [case], "langgraph")

    def test_validate_bad_tokens_raises(self):
        """validate_phase3_results raises when tokens_used != 0."""
        from aml_copilot.phase3_compare._shared import validate_phase3_results
        from aml_copilot.schemas import EvalCase, Phase3CaseResult

        result = Phase3CaseResult(
            framework="langgraph", case_id="C1", account_id="A",
            disposition="CLEAR", decision_reason="clear",
            sanctions_hits=[], rule_firings=[], anomaly_score=None,
            latency_ms=1.0, agent_reasoning="", agent_override=False,
            baseline_disposition="CLEAR", human_review_flagged=False,
            tokens_used=3,
        )
        case = EvalCase(case_id="C1", account_id="A", gold_label="CLEAR",
                        case_type="ibm_labeled", relevant_txn_ids=[], notes="")
        with pytest.raises(RuntimeError, match="tokens"):
            validate_phase3_results([result], [case], "langgraph")

    def test_validate_bad_cost_raises(self):
        """validate_phase3_results raises when cost_usd != 0.0."""
        from aml_copilot.phase3_compare._shared import validate_phase3_results
        from aml_copilot.schemas import EvalCase, Phase3CaseResult

        result = Phase3CaseResult(
            framework="langgraph", case_id="C1", account_id="A",
            disposition="CLEAR", decision_reason="clear",
            sanctions_hits=[], rule_firings=[], anomaly_score=None,
            latency_ms=1.0, agent_reasoning="", agent_override=False,
            baseline_disposition="CLEAR", human_review_flagged=False,
            cost_usd=0.01,
        )
        case = EvalCase(case_id="C1", account_id="A", gold_label="CLEAR",
                        case_type="ibm_labeled", relevant_txn_ids=[], notes="")
        with pytest.raises(RuntimeError, match="cost"):
            validate_phase3_results([result], [case], "langgraph")
