from __future__ import annotations

import pytest

from aml_copilot.schemas import AnomalyScore, RuleFiring, SanctionsHit


def test_sanctions_90_always_escalates(
    sample_sanctions_hit, sample_rule_firing, sample_anomaly_score
) -> None:
    """SanctionsHit.score >= 0.90 → ESCALATE regardless of rules and anomaly."""
    ...


def test_severity3_alone_escalates(sample_rule_firing) -> None:
    """Severity-3 RuleFiring with no sanctions and no anomaly → ESCALATE."""
    ...


def test_anomaly_alone_clears(sample_anomaly_score) -> None:
    """AnomalyScore.is_flagged == True with no rule >= severity 2 → CLEAR."""
    ...


def test_anomaly_plus_severity2_escalates(
    sample_anomaly_score, sample_rule_firing
) -> None:
    """Anomaly flagged AND at least one severity-2 rule → ESCALATE."""
    ...


def test_all_90_produce_valid_result(tmp_path) -> None:
    """Running against full eval.jsonl produces exactly 90 CaseResult rows, all schema-valid."""
    ...


def test_single_command_exits_0(tmp_path) -> None:
    """run_baseline.py invocation with valid --eval and --out exits with code 0."""
    ...
