"""
Phase 3 M5 — metrics computation for the cross-framework comparison.

Public API
----------
compute_framework_metrics(results, eval_cases, runner_file) -> Phase3FrameworkMetrics
check_framework_agreement(framework_results)               -> dict[str, Any]
compute_comparison_metrics(...)                             -> Phase3ComparisonMetrics

All functions are pure (no I/O, no side effects).  The comparison runner
(run_comparison.py) owns I/O and calls these functions.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from aml_copilot.phase3_compare.protocol import PROTOCOL_VERSION
from aml_copilot.schemas import (
    EvalCase,
    Phase3CaseResult,
    Phase3ComparisonMetrics,
    Phase3FrameworkMetrics,
)

# ── Severity weighting (mirrors step8_metrics._weight) ───────────────────────
# Replicated here to keep phase3_compare independent of step8_metrics.

_SEVERITY_WEIGHTS: dict[int, float] = {1: 1.0, 2: 2.0, 3: 3.0}


def _case_weight(case: EvalCase) -> float:
    """Return severity weight for a gold-ESCALATE EvalCase."""
    if case.severity_band is not None:
        return _SEVERITY_WEIGHTS[int(case.severity_band)]
    if case.case_type == "sanctions_hit":
        return 3.0
    if (
        case.case_type == "rules_anomaly_conflict"
        and case.conflict_type == "rule3_no_anomaly"
    ):
        return 3.0
    if case.case_type == "typology":
        return 2.0
    return 1.0


# ── LOC helper ────────────────────────────────────────────────────────────────


def count_loc(path: Path) -> int:
    """Count non-blank lines in a Python source file (physical LOC - blank lines)."""
    text = path.read_text(encoding="utf-8")
    return sum(1 for line in text.splitlines() if line.strip())


# ── Per-framework metrics ─────────────────────────────────────────────────────


def compute_framework_metrics(
    results: list[Phase3CaseResult],
    eval_cases: list[EvalCase],
    runner_file: Path,
) -> Phase3FrameworkMetrics:
    """Compute Phase3FrameworkMetrics for one framework's result set.

    Parameters
    ----------
    results : list[Phase3CaseResult]
        All 90 case results from this framework, in eval-file order.
    eval_cases : list[EvalCase]
        The frozen eval set (gold labels and case metadata).
    runner_file : Path
        Path to the primary Phase 3 runner source file (for LOC count).

    Returns
    -------
    Phase3FrameworkMetrics
        Fully populated metrics object.

    Raises
    ------
    ValueError
        If results is empty, or if any case_id in results is absent from eval_cases.
    """
    if not results:
        raise ValueError("results list is empty")

    framework = results[0].framework
    n = len(results)

    gold_map: dict[str, EvalCase] = {c.case_id: c for c in eval_cases}

    # ── Disposition accuracy ──────────────────────────────────────────────────
    correct = sum(
        1 for r in results
        if gold_map.get(r.case_id) is not None
        and gold_map[r.case_id].gold_label == r.disposition
    )
    accuracy = correct / n

    # ── Weighted false-clear rate ─────────────────────────────────────────────
    escalate_cases = [c for c in eval_cases if c.gold_label == "ESCALATE"]
    result_map: dict[str, Phase3CaseResult] = {r.case_id: r for r in results}

    weighted_denom = sum(_case_weight(c) for c in escalate_cases)
    weighted_fn = sum(
        _case_weight(c)
        for c in escalate_cases
        if result_map.get(c.case_id) is not None
        and result_map[c.case_id].disposition == "CLEAR"
    )
    fcr_weighted = weighted_fn / weighted_denom if weighted_denom > 0 else 0.0

    # ── Override rate ─────────────────────────────────────────────────────────
    overrides = sum(1 for r in results if r.agent_override)
    override_rate = overrides / n

    # ── Human review rate ─────────────────────────────────────────────────────
    reviews = sum(1 for r in results if r.human_review_flagged)
    human_review_rate = reviews / n

    # ── Latency ───────────────────────────────────────────────────────────────
    lats = np.array([r.latency_ms for r in results], dtype=np.float64)
    p50 = float(np.percentile(lats, 50))
    p95 = float(np.percentile(lats, 95))
    avg_lat = float(lats.mean())
    min_lat = float(lats.min())
    max_lat = float(lats.max())

    # ── Cost / token verification ─────────────────────────────────────────────
    zero_cost = all(r.cost_usd == 0.0 for r in results)
    zero_tokens = all(r.tokens_used == 0 for r in results)
    total_cost = sum(r.cost_usd for r in results)

    # ── LOC ───────────────────────────────────────────────────────────────────
    loc = count_loc(runner_file) if runner_file.exists() else 0

    return Phase3FrameworkMetrics(
        framework=framework,
        disposition_accuracy=accuracy,
        false_clear_rate_weighted=fcr_weighted,
        override_rate=override_rate,
        human_review_rate=human_review_rate,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        average_latency_ms=avg_lat,
        minimum_latency_ms=min_lat,
        maximum_latency_ms=max_lat,
        case_count=n,
        zero_cost_verified=zero_cost,
        zero_tokens_verified=zero_tokens,
        loc=loc,
        total_cost_usd=total_cost,
        eval_size=n,
    )


# ── Cross-framework agreement ─────────────────────────────────────────────────


def check_framework_agreement(
    framework_results: dict[str, list[Phase3CaseResult]],
) -> dict[str, Any]:
    """Verify field-by-field equality across all framework result sets.

    Checks equality of:
      disposition, decision_reason, agent_reasoning, human_review_flagged,
      agent_override, cost_usd, tokens_used, case ordering, case ids.

    Parameters
    ----------
    framework_results : dict[str, list[Phase3CaseResult]]
        Keys are framework names; values are result lists in eval-file order.
        Must contain at least two frameworks to be meaningful.

    Returns
    -------
    dict with keys
        all_dispositions_agree          : bool
        all_reasoning_agree             : bool  (decision_reason AND agent_reasoning)
        all_human_review_flags_agree    : bool
        all_overrides_agree             : bool
        case_ids_agree                  : bool
        case_ordering_agree             : bool
        all_costs_zero                  : bool
        all_tokens_zero                 : bool
        disposition_disagreements       : list[(case_id, {fw: val})]
        reasoning_disagreements         : list[(case_id, field, {fw: val})]
        human_review_disagreements      : list[(case_id, {fw: val})]
        override_disagreements          : list[(case_id, {fw: val})]
        cost_errors                     : list[(fw, case_id, val)]
        token_errors                    : list[(fw, case_id, val)]
    """
    if not framework_results:
        return _empty_agreement()

    frameworks = list(framework_results.keys())
    result_lists = list(framework_results.values())
    reference_name = frameworks[0]
    reference = result_lists[0]

    # ── Case ID and ordering agreement ───────────────────────────────────────
    ref_ids = [r.case_id for r in reference]
    case_ids_agree = True
    case_ordering_agree = True
    for fw, res in zip(frameworks[1:], result_lists[1:]):
        other_ids = [r.case_id for r in res]
        if set(other_ids) != set(ref_ids):
            case_ids_agree = False
        if other_ids != ref_ids:
            case_ordering_agree = False

    # ── Build per-framework maps for aligned comparison ───────────────────────
    fw_maps: dict[str, dict[str, Phase3CaseResult]] = {
        fw: {r.case_id: r for r in res}
        for fw, res in framework_results.items()
    }

    disposition_disagreements: list[tuple] = []
    reasoning_disagreements: list[tuple] = []
    human_review_disagreements: list[tuple] = []
    override_disagreements: list[tuple] = []
    cost_errors: list[tuple] = []
    token_errors: list[tuple] = []

    for ref_result in reference:
        cid = ref_result.case_id

        # Build per-framework values for this case
        disp_vals = {fw: fw_maps[fw][cid].disposition for fw in frameworks if cid in fw_maps[fw]}
        if len(set(disp_vals.values())) > 1:
            disposition_disagreements.append((cid, disp_vals))

        for field in ("decision_reason", "agent_reasoning"):
            field_vals = {fw: getattr(fw_maps[fw][cid], field) for fw in frameworks if cid in fw_maps[fw]}
            if len(set(field_vals.values())) > 1:
                reasoning_disagreements.append((cid, field, field_vals))

        hr_vals = {fw: fw_maps[fw][cid].human_review_flagged for fw in frameworks if cid in fw_maps[fw]}
        if len(set(hr_vals.values())) > 1:
            human_review_disagreements.append((cid, hr_vals))

        ov_vals = {fw: fw_maps[fw][cid].agent_override for fw in frameworks if cid in fw_maps[fw]}
        if len(set(ov_vals.values())) > 1:
            override_disagreements.append((cid, ov_vals))

    # ── Cost / token checks (across all frameworks, all cases) ───────────────
    for fw, res in framework_results.items():
        for r in res:
            if r.cost_usd != 0.0:
                cost_errors.append((fw, r.case_id, r.cost_usd))
            if r.tokens_used != 0:
                token_errors.append((fw, r.case_id, r.tokens_used))

    return {
        "all_dispositions_agree":         len(disposition_disagreements) == 0,
        "all_reasoning_agree":            len(reasoning_disagreements) == 0,
        "all_human_review_flags_agree":   len(human_review_disagreements) == 0,
        "all_overrides_agree":            len(override_disagreements) == 0,
        "case_ids_agree":                 case_ids_agree,
        "case_ordering_agree":            case_ordering_agree,
        "all_costs_zero":                 len(cost_errors) == 0,
        "all_tokens_zero":                len(token_errors) == 0,
        "disposition_disagreements":      disposition_disagreements,
        "reasoning_disagreements":        reasoning_disagreements,
        "human_review_disagreements":     human_review_disagreements,
        "override_disagreements":         override_disagreements,
        "cost_errors":                    cost_errors,
        "token_errors":                   token_errors,
    }


def _empty_agreement() -> dict[str, Any]:
    return {
        "all_dispositions_agree": True,
        "all_reasoning_agree": True,
        "all_human_review_flags_agree": True,
        "all_overrides_agree": True,
        "case_ids_agree": True,
        "case_ordering_agree": True,
        "all_costs_zero": True,
        "all_tokens_zero": True,
        "disposition_disagreements": [],
        "reasoning_disagreements": [],
        "human_review_disagreements": [],
        "override_disagreements": [],
        "cost_errors": [],
        "token_errors": [],
    }


# ── Comparison metrics assembly ───────────────────────────────────────────────


def compute_comparison_metrics(
    framework_results: dict[str, list[Phase3CaseResult]],
    framework_metrics: list[Phase3FrameworkMetrics],
    eval_cases: list[EvalCase],
    phase1_accuracy: float,
    phase2_accuracy: float,
    framework_version_information: dict[str, str] | None = None,
    runner_errors: dict[str, Exception] | None = None,
) -> Phase3ComparisonMetrics:
    """Build the cross-framework Phase3ComparisonMetrics report.

    Parameters
    ----------
    framework_results : dict[str, list[Phase3CaseResult]]
        Results from each successfully executed framework.
    framework_metrics : list[Phase3FrameworkMetrics]
        Per-framework metrics in registry order.
    eval_cases : list[EvalCase]
        Gold-label eval set.
    phase1_accuracy : float
        Disposition accuracy from the Phase 1 deterministic baseline.
    phase2_accuracy : float
        Disposition accuracy from the Phase 2 LangGraph evaluation.
    framework_version_information : dict[str, str], optional
        Package version strings keyed by framework name.
    runner_errors : dict[str, Exception], optional
        Frameworks that raised during run(); if any, comparison_passed=False.

    Returns
    -------
    Phase3ComparisonMetrics
    """
    agreement = check_framework_agreement(framework_results)

    any_runner_failed = bool(runner_errors)
    comparison_passed = (
        not any_runner_failed
        and agreement["all_dispositions_agree"]
        and agreement["all_reasoning_agree"]
        and agreement["all_human_review_flags_agree"]
        and agreement["all_overrides_agree"]
        and agreement["case_ids_agree"]
        and agreement["case_ordering_agree"]
        and agreement["all_costs_zero"]
        and agreement["all_tokens_zero"]
    )

    return Phase3ComparisonMetrics(
        generated_at=datetime.now(tz=timezone.utc),
        eval_size=len(eval_cases),
        protocol_version=PROTOCOL_VERSION,
        framework_version_information=framework_version_information or {},
        phase1_accuracy=phase1_accuracy,
        phase2_accuracy=phase2_accuracy,
        frameworks=framework_metrics,
        all_dispositions_agree=agreement["all_dispositions_agree"],
        all_reasoning_agree=agreement["all_reasoning_agree"],
        all_human_review_flags_agree=agreement["all_human_review_flags_agree"],
        all_costs_zero=agreement["all_costs_zero"],
        all_tokens_zero=agreement["all_tokens_zero"],
        comparison_passed=comparison_passed,
    )


# ── Framework version discovery ───────────────────────────────────────────────


def get_framework_versions() -> dict[str, str]:
    """Return installed version strings for all Phase 3 framework dependencies."""
    import importlib.metadata

    pkg_map = {
        "python": None,          # special case handled below
        "langgraph": "langgraph",
        "crewai": "crewai",
        "openai_agents": "openai-agents",
    }
    versions: dict[str, str] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    for key, pkg in pkg_map.items():
        if key == "python" or pkg is None:
            continue
        try:
            versions[key] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "unknown"
    return versions
