from __future__ import annotations

from aml_copilot.schemas import AnomalyScore, CaseResult, RuleFiring, SanctionsHit

# Fixed precedence — do not modify after baseline freeze.
# Changing these thresholds after seeing eval metrics is equivalent to eval leakage.
SANCTIONS_ESCALATION_THRESHOLD: float = 0.90
CRITICAL_RULE_SEVERITY: int = 3
ELEVATED_RULE_SEVERITY: int = 2


def apply_decision_table(
    case_id: str,
    account_id: str,
    sanctions_hits: list[SanctionsHit],
    rule_firings: list[RuleFiring],
    anomaly_score: AnomalyScore | None,
    latency_ms: float,
) -> CaseResult:
    """
    Fixed-precedence decision table:
      1. sanctions hit >= 0.90 OR severity-3 rule  → ESCALATE
      2. anomaly flagged AND severity >= 2 rule     → ESCALATE
      3. else                                       → CLEAR
    """
    ...
