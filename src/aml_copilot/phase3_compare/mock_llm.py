"""
Deterministic mock LLM — single source of truth for the Phase 2–3 decision policy.

All three Phase 3 framework runners (LangGraph, CrewAI, OpenAI Agents SDK) import
mock_llm_call from here.  The function body is copied verbatim from the Phase 2
_mock_llm_call; no branch, threshold, confidence value, or reasoning string has
been altered.

phase2_eval.run_langgraph_eval re-exports mock_llm_call as _mock_llm_call and
re-exports all constants so existing Phase 2 tests continue to import them from
their original location.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

# ── Decision policy thresholds ────────────────────────────────────────────────
# Not frozen — they exist to demonstrate added value over the Phase 1 decision
# table.  Moving them here gives all Phase 3 runners a single reference point.

AGENT_ANOMALY_PCT_THRESHOLD: float = 0.90      # Branch 3 escalation lower bound
AGENT_MIN_RULE_SEV_FOR_OVERRIDE: int = 2       # minimum rule severity for Branch 3
HUMAN_REVIEW_ANOMALY_THRESHOLD: float = 0.85  # Branch 4 human-review flag threshold

# Phase 3 naming aliases (introduced in M1; values are identical to the originals).
# HUMAN_REVIEW_ANOMALY_MIN  = lower bound of the "elevated but not escalated" zone
# HUMAN_REVIEW_ANOMALY_MAX  = upper bound; above this + elevated rule → escalate
HUMAN_REVIEW_ANOMALY_MIN: float = HUMAN_REVIEW_ANOMALY_THRESHOLD   # 0.85
HUMAN_REVIEW_ANOMALY_MAX: float = AGENT_ANOMALY_PCT_THRESHOLD        # 0.90


# ── Output type ───────────────────────────────────────────────────────────────

class MockLLMOutput(TypedDict):
    disposition: Literal["ESCALATE", "CLEAR"]
    decision_reason: str
    reasoning: str
    confidence: float
    human_review: bool


# ── Public function ───────────────────────────────────────────────────────────

def mock_llm_call(evidence: dict[str, Any]) -> MockLLMOutput:
    """
    Deterministic mock LLM — no API calls, no random state.

    Parameters
    ----------
    evidence : dict
        Keys consumed (all optional; missing keys default gracefully):
          "sanctions_hits"  list[dict]  each dict must contain "match_score": float
          "rule_firings"    list[dict]  each dict must contain "severity": int
                                        and "rule_id": str
          "anomaly_score"   dict | None  must contain "percentile": float
                                         and "score": float when not None

    Returns
    -------
    MockLLMOutput
        Always contains all five keys: disposition, decision_reason,
        reasoning, confidence, human_review.

    Decision branches (evaluated in order; first match wins)
    --------------------------------------------------------
    Branch 1 — sanctions hit   : max sanctions score >= 0.90  → ESCALATE
    Branch 2 — critical rule   : max rule severity == 3       → ESCALATE
    Branch 3 — agent extension : anomaly_pct >= 0.90
                                 AND max_rule_sev >= 2        → ESCALATE
    Branch 4 — clear           : fall-through; human_review
                                 flagged if anomaly_pct > 0.85
    """
    sanctions_hits = evidence.get("sanctions_hits", [])
    rule_firings = evidence.get("rule_firings", [])
    anomaly = evidence.get("anomaly_score")

    max_sanctions_score = max(
        (h["match_score"] for h in sanctions_hits), default=0.0
    )
    max_rule_sev = max((f["severity"] for f in rule_firings), default=0)
    anomaly_pct = anomaly["percentile"] if anomaly else 0.0
    anomaly_score_val = anomaly["score"] if anomaly else 0.0

    # Branch 1: sanctions hit — score >= 0.90
    if max_sanctions_score >= 0.90:
        return {
            "disposition": "ESCALATE",
            "decision_reason": "sanctions_or_critical_rule",
            "reasoning": (
                f"Sanctions hit score {max_sanctions_score:.3f} ≥ 0.90 threshold. "
                "OFAC compliance obligation requires immediate escalation."
            ),
            "confidence": 0.99,
            "human_review": False,
        }

    # Branch 2: critical rule — severity-3 firing
    if max_rule_sev == 3:
        sev3_ids = [f["rule_id"] for f in rule_firings if f["severity"] == 3]
        return {
            "disposition": "ESCALATE",
            "decision_reason": "sanctions_or_critical_rule",
            "reasoning": (
                f"Critical severity-3 rule(s) fired: {sev3_ids}. "
                "High-risk pattern warrants escalation."
            ),
            "confidence": 0.95,
            "human_review": False,
        }

    # Branch 3: agent extension — anomaly >= 0.90 + severity-2 rule
    if (
        anomaly_pct >= AGENT_ANOMALY_PCT_THRESHOLD
        and max_rule_sev >= AGENT_MIN_RULE_SEV_FOR_OVERRIDE
    ):
        return {
            "disposition": "ESCALATE",
            "decision_reason": "agent_anomaly_plus_elevated_rule",
            "reasoning": (
                f"Anomaly at {anomaly_pct:.1%} percentile "
                f"(robust-z {anomaly_score_val:.2f}) — top "
                f"{(1 - anomaly_pct) * 100:.1f}% of account population — "
                f"combined with severity-{max_rule_sev} rule. "
                "Deterministic baseline requires the 99.5th-percentile flag; "
                "agent escalates at the 90th-percentile when paired with "
                "elevated rule evidence."
            ),
            "confidence": 0.78,
            "human_review": True,
        }

    # Branch 4: default CLEAR; flag for human review if anomaly is elevated
    human_review = anomaly_pct > HUMAN_REVIEW_ANOMALY_THRESHOLD
    if human_review:
        reasoning = (
            f"Anomaly at {anomaly_pct:.1%} percentile with max rule severity "
            f"{max_rule_sev}. Below escalation threshold but elevated; "
            "flagging for human review."
        )
    else:
        reasoning = (
            f"No significant risk indicators: anomaly {anomaly_pct:.1%} "
            f"percentile, max rule severity {max_rule_sev}. Case clears."
        )

    return {
        "disposition": "CLEAR",
        "decision_reason": "clear",
        "reasoning": reasoning,
        "confidence": 0.88,
        "human_review": human_review,
    }
