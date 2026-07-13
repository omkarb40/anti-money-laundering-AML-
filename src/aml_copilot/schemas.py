from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel


class Transaction(BaseModel):
    txn_id: str
    timestamp: datetime
    from_account: str
    to_account: str
    amount: float
    payment_type: str
    is_laundering: bool


class Account(BaseModel):
    account_id: str
    name: str
    country: Optional[str] = None
    kyc_risk: Optional[str] = None  # "low" | "medium" | "high"


class GroundTruthRow(BaseModel):
    row_id: int
    account_id: str
    assigned_name: str
    ofac_uid: str
    ofac_canonical_name: str  # screening reference only — never user-facing
    match_flavor: Literal[
        "exact", "transliteration", "typo_ocr", "partial_reorder", "hard_negative"
    ]
    expected_score_min: float
    expected_score_max: float
    gold_is_match: bool


class SanctionsHit(BaseModel):
    account_id: str
    assigned_name: str  # internal evidence only — never logged/printed
    ofac_uid: str
    list_source: Literal["SDN", "Consolidated"]
    match_score: float  # max(jaro_winkler, token_sort_ratio / 100)
    scorer_used: Literal["exact", "jaro_winkler", "token_sort_ratio"]
    matched_name_type: Literal["canonical", "alias"]


class EntityChain(BaseModel):
    account_id: str
    name: str
    country: Optional[str] = None
    kyc_risk: Optional[str] = None
    hop1_counterparties: list[str]
    hop2_counterparties: list[str]  # capped at 50
    pattern_label: Optional[str] = None


class RuleFiring(BaseModel):
    rule_id: str
    severity: Literal[1, 2, 3]
    account_id: str
    evidence: dict[str, Any]
    window_start: datetime
    window_end: datetime


class AnomalyScore(BaseModel):
    account_id: str
    score: float
    percentile: float
    is_flagged: bool
    excluded_features: list[str]


class EvalCase(BaseModel):
    case_id: str
    account_id: str
    gold_label: Literal["ESCALATE", "CLEAR"]
    case_type: Literal[
        "ibm_labeled",
        "sanctions_hit",
        "sanctions_near_miss",
        "rules_anomaly_conflict",
        "typology",
    ]
    typology: Optional[str] = None
    severity_band: Optional[Literal[1, 2, 3]] = None  # set for ibm_labeled cases
    conflict_type: Optional[Literal["rule_no_anomaly", "anomaly_no_rule", "rule3_no_anomaly"]] = None
    relevant_txn_ids: list[str]
    notes: str


class CaseResult(BaseModel):
    case_id: str
    account_id: str
    disposition: Literal["ESCALATE", "CLEAR"]
    decision_reason: str
    sanctions_hits: list[SanctionsHit]
    rule_firings: list[RuleFiring]
    anomaly_score: Optional[AnomalyScore] = None
    latency_ms: float


class MetricsReport(BaseModel):
    disposition_accuracy: float
    false_clear_rate_weighted: float  # primary metric; SAR misses cost most
    sanctions_precision: float
    sanctions_recall: float
    latency_p50_ms: float
    latency_p95_ms: float
    total_cost_usd: float  # must be 0.0 for Phase 1–3
    eval_size: int
    generated_at: datetime


# ── Phase 3 ──────────────────────────────────────────────────────────────────
# Phase3CaseResult is defined flat (not inheriting Phase2CaseResult) to keep
# schemas.py free of imports from phase2_eval.  It carries all Phase2CaseResult
# fields plus the framework identifier added in Phase 3.

class Phase3CaseResult(BaseModel):
    """Per-case output produced by any Phase 3 framework runner."""
    case_id: str
    account_id: str
    framework: str                                  # "langgraph" | "crewai" | "openai_agents"
    disposition: Literal["ESCALATE", "CLEAR"]
    decision_reason: str
    sanctions_hits: list[SanctionsHit]
    rule_firings: list[RuleFiring]
    anomaly_score: Optional[AnomalyScore] = None
    latency_ms: float
    agent_reasoning: str
    agent_override: bool                            # True iff disposition != baseline_disposition
    baseline_disposition: Literal["ESCALATE", "CLEAR"]
    human_review_flagged: bool
    tokens_used: int = 0                            # always 0 in mock mode
    cost_usd: float = 0.0                           # always 0.0 in mock mode


class Phase3FrameworkMetrics(BaseModel):
    """Aggregate metrics for one framework in the Phase 3 comparison."""
    framework: str
    disposition_accuracy: float
    false_clear_rate_weighted: float
    override_rate: float                            # fraction where disposition != Phase 1 baseline
    human_review_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    average_latency_ms: float = 0.0
    minimum_latency_ms: float = 0.0
    maximum_latency_ms: float = 0.0
    case_count: int = 0
    zero_cost_verified: bool = True
    zero_tokens_verified: bool = True
    loc: int                                        # non-blank lines in the primary Phase 3 runner file
    total_cost_usd: float                           # always 0.0 in mock mode
    eval_size: int


class Phase3ComparisonMetrics(BaseModel):
    """Cross-framework comparison report produced by the M5 comparison runner."""
    generated_at: datetime
    eval_size: int
    protocol_version: str                           # matches PROTOCOL_VERSION in phase3_compare.protocol
    framework_version_information: dict[str, str] = {}
    phase1_accuracy: float                          # from artifacts/metrics_baseline.json
    phase2_accuracy: float                          # from artifacts/phase2_langgraph_metrics.json
    frameworks: list[Phase3FrameworkMetrics]        # ordered: langgraph, crewai, openai_agents
    all_dispositions_agree: bool
    all_reasoning_agree: bool                       # decision_reason + agent_reasoning identical
    all_human_review_flags_agree: bool
    all_costs_zero: bool
    all_tokens_zero: bool
    comparison_passed: bool                         # True iff every agreement check passes
