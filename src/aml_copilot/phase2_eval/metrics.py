"""
Phase 2.5 — metrics computation and baseline comparison.

compute_phase2_metrics()  — pure function; no I/O.
save_phase2_metrics()     — writes artifacts/phase2_langgraph_metrics.json.
main()                    — CLI; reads phase2_langgraph_results.jsonl and
                            metrics_baseline.json, prints comparison table.

These artifacts are NOT frozen (unlike the Phase 1–3 baseline).  Phase 4 will
produce the definitive LLM baseline; Phase 2.5 is exploratory.

Metric definitions
------------------
Shared with Step 8 (same formulas):
  disposition_accuracy, false_clear_rate_weighted,
  sanctions_precision, sanctions_recall, latency_p50/p95_ms

Phase 2.5 additions:
  total_tokens_used    int    sum of tokens_used across all cases (0 in mock)
  total_cost_usd       float  sum of cost_usd (0.0 in mock)
  human_review_rate    float  fraction of cases flagged for human review
  override_rate        float  fraction where agent changed baseline disposition
  baseline_disposition_accuracy  float  from metrics_baseline.json
  baseline_false_clear_rate      float  from metrics_baseline.json
  delta_accuracy       float  phase2 accuracy − baseline accuracy
  delta_false_clear_rate float phase2 FCR − baseline FCR  (negative = improvement)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from aml_copilot.schemas import CaseResult, EvalCase, MetricsReport
from aml_copilot.step8_metrics.metrics import compute_metrics
from aml_copilot.phase2_eval.run_langgraph_eval import Phase2CaseResult

logger = logging.getLogger(__name__)

_ROOT               = Path(__file__).parents[3]
_PHASE2_RESULTS     = _ROOT / "artifacts/phase2_langgraph_results.jsonl"
_BASELINE_METRICS   = _ROOT / "artifacts/metrics_baseline.json"
_EVAL_PATH          = _ROOT / "data/fixtures/eval.jsonl"
_OUT_PATH           = _ROOT / "artifacts/phase2_langgraph_metrics.json"


# ── Phase2 metrics schema ─────────────────────────────────────────────────────

class Phase2MetricsReport(BaseModel):
    """Metrics for the Phase 2.5 LangGraph agent run."""
    eval_size: int
    generated_at: datetime
    # Accuracy (same formulas as Step 8)
    disposition_accuracy: float
    false_clear_rate_weighted: float
    sanctions_precision: float
    sanctions_recall: float
    # Latency
    latency_p50_ms: float
    latency_p95_ms: float
    # LLM usage (zero in mock mode)
    total_tokens_used: int
    total_cost_usd: float
    # Phase 2.5-specific
    human_review_rate: float
    override_rate: float
    # Baseline comparison
    baseline_disposition_accuracy: float
    baseline_false_clear_rate: float
    delta_accuracy: float          # phase2 − baseline  (positive = improvement)
    delta_false_clear_rate: float  # phase2 FCR − baseline FCR (negative = better)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_phase2_metrics(
    results: list[Phase2CaseResult],
    eval_cases: list[EvalCase],
    baseline_report: MetricsReport,
) -> Phase2MetricsReport:
    """
    Compute all Phase 2.5 metrics.

    Parameters
    ----------
    results : list[Phase2CaseResult]
        Output of run_langgraph_eval.run().
    eval_cases : list[EvalCase]
        Frozen eval set from data/fixtures/eval.jsonl.
    baseline_report : MetricsReport
        Frozen baseline from artifacts/metrics_baseline.json.

    Returns
    -------
    Phase2MetricsReport
    """
    if not results:
        raise ValueError("results is empty")
    if not eval_cases:
        raise ValueError("eval_cases is empty")

    # Convert Phase2CaseResult → CaseResult so we can reuse Step 8's
    # compute_metrics() directly (avoids duplicating metric formulas).
    case_results = [
        CaseResult(
            case_id=r.case_id,
            account_id=r.account_id,
            disposition=r.disposition,
            decision_reason=r.decision_reason,
            sanctions_hits=r.sanctions_hits,
            rule_firings=r.rule_firings,
            anomaly_score=r.anomaly_score,
            latency_ms=r.latency_ms,
        )
        for r in results
    ]
    base = compute_metrics(case_results, eval_cases)

    n = len(results)
    human_review_rate = sum(1 for r in results if r.human_review_flagged) / n
    override_rate = sum(1 for r in results if r.agent_override) / n
    total_tokens = sum(r.tokens_used for r in results)
    total_cost = sum(r.cost_usd for r in results)

    return Phase2MetricsReport(
        eval_size=n,
        generated_at=datetime.now(tz=timezone.utc),
        disposition_accuracy=base.disposition_accuracy,
        false_clear_rate_weighted=base.false_clear_rate_weighted,
        sanctions_precision=base.sanctions_precision,
        sanctions_recall=base.sanctions_recall,
        latency_p50_ms=base.latency_p50_ms,
        latency_p95_ms=base.latency_p95_ms,
        total_tokens_used=total_tokens,
        total_cost_usd=total_cost,
        human_review_rate=human_review_rate,
        override_rate=override_rate,
        baseline_disposition_accuracy=baseline_report.disposition_accuracy,
        baseline_false_clear_rate=baseline_report.false_clear_rate_weighted,
        delta_accuracy=base.disposition_accuracy - baseline_report.disposition_accuracy,
        delta_false_clear_rate=(
            base.false_clear_rate_weighted - baseline_report.false_clear_rate_weighted
        ),
    )


def save_phase2_metrics(
    report: Phase2MetricsReport,
    path: str | Path = _OUT_PATH,
) -> None:
    """Write Phase2MetricsReport to path as indented JSON (not checksummed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)
    logger.info("[Phase2] metrics written: %s", path)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_phase2_results(path: Path) -> list[Phase2CaseResult]:
    rows: list[Phase2CaseResult] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(Phase2CaseResult.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"phase2_langgraph_results.jsonl line {i}: {exc}"
                ) from exc
    return rows


def _load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(EvalCase.model_validate_json(line))
    return cases


def _load_baseline_metrics(path: Path) -> MetricsReport:
    return MetricsReport.model_validate_json(path.read_text(encoding="utf-8"))


def _print_comparison(report: Phase2MetricsReport) -> None:
    delta_sign = lambda v: (f"+{v:.4f}" if v >= 0 else f"{v:.4f}")
    improvement = lambda v: "↑ better" if v > 0 else ("↓ worse" if v < 0 else "—")
    fcr_improvement = lambda v: "↓ better" if v < 0 else ("↑ worse" if v > 0 else "—")

    print(f"\n{'=' * 70}")
    print(f"Phase 2.5 LangGraph Metrics  (eval_size={report.eval_size})")
    print(f"{'─' * 70}")
    print(f"{'Metric':<38}{'Phase2':>12}{'Baseline':>12}{'Δ':>8}")
    print(f"{'─' * 70}")
    print(
        f"{'Disposition accuracy':<38}"
        f"{report.disposition_accuracy:>12.4f}"
        f"{report.baseline_disposition_accuracy:>12.4f}"
        f"{delta_sign(report.delta_accuracy):>8}  {improvement(report.delta_accuracy)}"
    )
    print(
        f"{'False-clear rate (wtd) ← primary':<38}"
        f"{report.false_clear_rate_weighted:>12.4f}"
        f"{report.baseline_false_clear_rate:>12.4f}"
        f"{delta_sign(report.delta_false_clear_rate):>8}  {fcr_improvement(report.delta_false_clear_rate)}"
    )
    print(f"{'Sanctions precision':<38}{report.sanctions_precision:>12.4f}")
    print(f"{'Sanctions recall':<38}{report.sanctions_recall:>12.4f}")
    print(f"{'─' * 70}")
    print(f"{'Human review rate':<38}{report.human_review_rate:>12.1%}")
    print(f"{'Override rate (vs baseline)':<38}{report.override_rate:>12.1%}")
    print(f"{'─' * 70}")
    print(f"{'Latency p50 (ms)':<38}{report.latency_p50_ms:>12.2f}")
    print(f"{'Latency p95 (ms)':<38}{report.latency_p95_ms:>12.2f}")
    print(f"{'Total tokens':<38}{report.total_tokens_used:>12d}")
    print(f"{'Total cost (USD)':<38}${report.total_cost_usd:>11.2f}")
    print(f"{'=' * 70}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2.5 metrics — compare LangGraph agent vs deterministic baseline"
    )
    p.add_argument("--results",  default=str(_PHASE2_RESULTS))
    p.add_argument("--eval",     default=str(_EVAL_PATH))
    p.add_argument("--baseline", default=str(_BASELINE_METRICS))
    p.add_argument("--out",      default=str(_OUT_PATH))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    for label, fpath in [
        ("phase2 results", args.results),
        ("eval",           args.eval),
        ("baseline metrics", args.baseline),
    ]:
        if not Path(fpath).exists():
            print(f"[FAIL] {label} not found: {fpath}", file=sys.stderr)
            sys.exit(1)

    logger.info("[Phase2] Loading phase2 results…")
    results = _load_phase2_results(Path(args.results))

    logger.info("[Phase2] Loading eval cases…")
    eval_cases = _load_eval_cases(Path(args.eval))

    logger.info("[Phase2] Loading baseline metrics…")
    baseline_report = _load_baseline_metrics(Path(args.baseline))

    logger.info("[Phase2] Computing metrics…")
    report = compute_phase2_metrics(results, eval_cases, baseline_report)

    save_phase2_metrics(report, path=Path(args.out))
    _print_comparison(report)
    print("[Phase2] metrics complete.")


if __name__ == "__main__":
    main()
