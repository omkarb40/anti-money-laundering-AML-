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
    candidate_name: str
    ofac_uid: str
    list_name: str  # "SDN" | "Consolidated"
    score: float
    match_type: Literal["exact", "transliteration", "typo_ocr", "partial_reorder"]


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
