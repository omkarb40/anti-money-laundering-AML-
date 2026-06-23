"""
Step 7: Fixed-precedence decision table.

apply_decision_table() is a pure function.  Given pre-computed tool outputs
for one account it returns a fully-populated CaseResult.

Branches (immutable — changing these after the baseline freeze is eval leakage):
  1. any SanctionsHit.match_score >= 0.90  OR  any RuleFiring.severity == 3
       → ESCALATE  reason="sanctions_or_critical_rule"
  2. anomaly_score is not None
       AND anomaly_score.is_flagged
       AND any RuleFiring.severity >= 2
       → ESCALATE  reason="anomaly_plus_elevated_rule"
  3. else
       → CLEAR  reason="clear"
"""
from __future__ import annotations

from aml_copilot.schemas import AnomalyScore, CaseResult, RuleFiring, SanctionsHit

# Fixed precedence — do not modify after baseline freeze.
# Changing thresholds after seeing eval metrics is equivalent to eval leakage.
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
    Fixed-precedence decision table — returns a fully-populated CaseResult.

    Exceptions propagate; there is no silent-CLEAR fallback for errors.
    """
    has_sanctions_trigger = any(
        h.match_score >= SANCTIONS_ESCALATION_THRESHOLD for h in sanctions_hits
    )
    has_critical_rule = any(
        f.severity == CRITICAL_RULE_SEVERITY for f in rule_firings
    )

    if has_sanctions_trigger or has_critical_rule:
        disposition: str = "ESCALATE"
        reason: str = "sanctions_or_critical_rule"
    elif (
        anomaly_score is not None
        and anomaly_score.is_flagged
        and any(f.severity >= ELEVATED_RULE_SEVERITY for f in rule_firings)
    ):
        disposition = "ESCALATE"
        reason = "anomaly_plus_elevated_rule"
    else:
        disposition = "CLEAR"
        reason = "clear"

    return CaseResult(
        case_id=case_id,
        account_id=account_id,
        disposition=disposition,
        decision_reason=reason,
        sanctions_hits=sanctions_hits,
        rule_firings=rule_firings,
        anomaly_score=anomaly_score,
        latency_ms=latency_ms,
    )
