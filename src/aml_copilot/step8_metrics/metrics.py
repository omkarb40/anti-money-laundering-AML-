"""
Step 8: Baseline metrics computation.

compute_metrics() is a pure function — no I/O, no side effects.
save_metrics()     writes artifacts/metrics_baseline.json and records its SHA-256.

The frozen metrics_baseline.json is the Phase 4 control row.  It must not be
modified after creation; save_metrics() enforces write-once semantics through
the shared checksum utility (raises RuntimeError on second write unless
force=True).

CLI:
    python -m aml_copilot.step8_metrics.metrics
    python -m aml_copilot.step8_metrics.metrics --force   # rebuild if frozen
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aml_copilot.schemas import CaseResult, EvalCase, MetricsReport
from aml_copilot.utils.checksum import append_checksum, compute_sha256, verify_checksums

logger = logging.getLogger(__name__)

_ROOT          = Path(__file__).parents[3]
_EVAL_PATH     = _ROOT / "data/fixtures/eval.jsonl"
_RESULTS_PATH  = _ROOT / "artifacts/results.jsonl"
_OUT_PATH      = _ROOT / "artifacts/metrics_baseline.json"
_CHECKSUM_FILE = _ROOT / "artifacts/checksums.sha256"

# Severity weights: a severity-3 false clear costs 3× a severity-1 false clear.
SEVERITY_WEIGHTS: dict[int, float] = {1: 1.0, 2: 2.0, 3: 3.0}


# ── Private helpers ───────────────────────────────────────────────────────────

def _weight(case: EvalCase) -> float:
    """Return the severity weight for an ESCALATE eval case.

    Priority order:
      ibm_labeled severity_band (explicit)     → 1 / 2 / 3
      sanctions_hit                             → 3 (Branch 1 critical)
      rules_anomaly_conflict / rule3_no_anomaly → 3 (severity-3 rule)
      typology                                  → 2 (pattern-identified)
      fallback                                  → 1
    """
    if case.severity_band is not None:
        return SEVERITY_WEIGHTS[int(case.severity_band)]
    if case.case_type == "sanctions_hit":
        return 3.0
    if case.case_type == "rules_anomaly_conflict" and case.conflict_type == "rule3_no_anomaly":
        return 3.0
    if case.case_type == "typology":
        return 2.0
    return 1.0


def _join(
    eval_cases: list[EvalCase],
    results: list[CaseResult],
) -> list[tuple[EvalCase, CaseResult]]:
    """
    Inner join eval_cases and results by case_id.

    Raises ValueError if:
      - any case_id in eval is absent from results
      - any case_id in results is absent from eval
      - any case_id appears more than once in results
    Returns pairs in sorted case_id order (deterministic).
    """
    dup_counts = Counter(r.case_id for r in results)
    dupes = sorted(cid for cid, n in dup_counts.items() if n > 1)
    if dupes:
        raise ValueError(f"Duplicate case_ids in results: {dupes}")

    eval_map   = {e.case_id: e for e in eval_cases}
    result_map = {r.case_id: r for r in results}

    missing = sorted(set(eval_map) - set(result_map))
    extra   = sorted(set(result_map) - set(eval_map))

    if missing:
        raise ValueError(f"Results missing case_ids from eval: {missing[:10]}")
    if extra:
        raise ValueError(f"Results contain case_ids not in eval: {extra[:10]}")

    return [(eval_map[cid], result_map[cid]) for cid in sorted(eval_map)]


def _remove_checksum_line(checksum_file: Path, path: Path) -> None:
    """Remove the checksum entry for path (used by --force).

    Uses the same repo-relative key logic as append_checksum so it correctly
    matches entries regardless of whether they were written in absolute or
    relative-path format.
    """
    if not checksum_file.exists():
        return
    from aml_copilot.utils.checksum import _to_key
    target = _to_key(path)
    lines = checksum_file.read_text(encoding="utf-8").splitlines(keepends=True)
    checksum_file.write_text(
        "".join(ln for ln in lines if target not in ln),
        encoding="utf-8",
    )


# ── Public API ────────────────────────────────────────────────────────────────

def compute_metrics(
    results: list[CaseResult],
    eval_cases: list[EvalCase],
) -> MetricsReport:
    """
    Compute all baseline metrics from results and gold labels.

    Parameters
    ----------
    results : list[CaseResult]
        Output of the Step 7 runner — one per eval case.
    eval_cases : list[EvalCase]
        Frozen eval set — gold labels and case metadata.

    Returns
    -------
    MetricsReport
        All metrics.  total_cost_usd is always 0.0 (no LLM calls in Phase 1–3).

    Raises
    ------
    ValueError
        If either list is empty, or if the case_id join fails.

    Notes
    -----
    Safe division: denominators that are 0 return 0.0 with a logged warning,
    not ZeroDivisionError.
    """
    if not results:
        raise ValueError("results is empty — pass at least one CaseResult")
    if not eval_cases:
        raise ValueError("eval_cases is empty — pass at least one EvalCase")

    pairs = _join(eval_cases, results)

    # ── Disposition accuracy ──────────────────────────────────────────────
    correct  = sum(1 for e, r in pairs if r.disposition == e.gold_label)
    accuracy = correct / len(pairs)

    # ── Weighted false-clear rate (primary metric) ────────────────────────
    escalate_pairs  = [(e, r) for e, r in pairs if e.gold_label == "ESCALATE"]
    weighted_denom  = sum(_weight(e) for e, r in escalate_pairs)
    weighted_fn     = sum(_weight(e) for e, r in escalate_pairs if r.disposition == "CLEAR")

    if weighted_denom == 0:
        logger.warning(
            "No ESCALATE gold cases — false_clear_rate_weighted is not meaningful; returning 0.0"
        )
        fcr_weighted = 0.0
    else:
        fcr_weighted = weighted_fn / weighted_denom

    # ── Sanctions precision and recall ────────────────────────────────────
    sh_tp  = sum(1 for e, r in pairs if e.case_type == "sanctions_hit"       and r.disposition == "ESCALATE")
    sh_fn  = sum(1 for e, r in pairs if e.case_type == "sanctions_hit"       and r.disposition == "CLEAR")
    snm_fp = sum(1 for e, r in pairs if e.case_type == "sanctions_near_miss" and r.disposition == "ESCALATE")

    if (sh_tp + snm_fp) == 0:
        logger.warning("No sanctions results to score — sanctions_precision returning 0.0")
        sanctions_precision = 0.0
    else:
        sanctions_precision = sh_tp / (sh_tp + snm_fp)

    if (sh_tp + sh_fn) == 0:
        logger.warning("No sanctions_hit eval cases — sanctions_recall returning 0.0")
        sanctions_recall = 0.0
    else:
        sanctions_recall = sh_tp / (sh_tp + sh_fn)

    # ── Latency ───────────────────────────────────────────────────────────
    latencies = np.array([r.latency_ms for _, r in pairs], dtype=np.float64)
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))

    return MetricsReport(
        disposition_accuracy=accuracy,
        false_clear_rate_weighted=fcr_weighted,
        sanctions_precision=sanctions_precision,
        sanctions_recall=sanctions_recall,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        total_cost_usd=0.0,
        eval_size=len(pairs),
        generated_at=datetime.now(tz=timezone.utc),
    )


def save_metrics(
    report: MetricsReport,
    path: str | Path = _OUT_PATH,
    checksums_path: str | Path = _CHECKSUM_FILE,
    force: bool = False,
) -> str:
    """
    Write MetricsReport to path as indented JSON, then record its SHA-256.

    Write-once: raises RuntimeError on second call with force=False (the entry
    already exists in checksums_path).  With force=True the existing checksum
    entry is removed before re-freezing.

    Parameters
    ----------
    report : MetricsReport
        Populated report from compute_metrics().
    path : path-like
        Destination file (default: artifacts/metrics_baseline.json).
    checksums_path : path-like
        Checksum index to append to (default: artifacts/checksums.sha256).
    force : bool
        If True, overwrite an existing frozen entry.

    Returns
    -------
    str
        SHA-256 hex digest of the written file.

    Raises
    ------
    AssertionError
        If report.total_cost_usd != 0.0 (Phase 1–3 must have zero LLM cost).
    RuntimeError
        If a checksum entry already exists and force=False.
    """
    assert report.total_cost_usd == 0.0, (
        f"Phase 1–3: total_cost_usd must be 0.0, got {report.total_cost_usd}. "
        "A non-zero cost indicates an LLM call was made."
    )
    assert report.generated_at.tzinfo is not None, (
        "generated_at must be UTC-aware (use datetime.now(tz=timezone.utc))"
    )

    path = Path(path)
    checksums_path = Path(checksums_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(".tmp")
    tmp.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)

    if force:
        _remove_checksum_line(checksums_path, path)

    append_checksum(path, checksums_path)
    return compute_sha256(path)


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _load_eval(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(EvalCase.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"eval.jsonl line {i}: {exc}") from exc
    return cases


def _load_results(path: Path) -> list[CaseResult]:
    rows: list[CaseResult] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(CaseResult.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"results.jsonl line {i}: {exc}") from exc
    return rows


def _print_summary(report: MetricsReport, path: Path, digest: str = "") -> None:
    print(f"\n{'=' * 64}")
    print(f"metrics_baseline.json: {path}  (eval_size={report.eval_size})")
    print(f"  Disposition accuracy:       {report.disposition_accuracy:.6f}")
    print(f"  False-clear rate (wtd):     {report.false_clear_rate_weighted:.6f}   ← PRIMARY")
    print(f"  Sanctions precision:        {report.sanctions_precision:.6f}")
    print(f"  Sanctions recall:           {report.sanctions_recall:.6f}")
    print(f"  Latency p50:                {report.latency_p50_ms:.3f} ms")
    print(f"  Latency p95:                {report.latency_p95_ms:.3f} ms")
    print(f"  Total cost:                 ${report.total_cost_usd:.2f}")
    print(f"  Generated at:               {report.generated_at.isoformat()}")
    if digest:
        print(f"  SHA-256:                    {digest}")
    print(f"{'=' * 64}\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute and freeze AML baseline metrics")
    p.add_argument("--results",   default=str(_RESULTS_PATH))
    p.add_argument("--eval",      default=str(_EVAL_PATH))
    p.add_argument("--checksums", default=str(_CHECKSUM_FILE))
    p.add_argument("--out",       default=str(_OUT_PATH))
    p.add_argument("--force",     action="store_true",
                   help="Re-freeze even if metrics_baseline.json already exists")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    out_path = Path(args.out)

    if out_path.exists() and not args.force:
        print(f"[SKIP] {out_path} already frozen. Use --force to rebuild.")
        report = MetricsReport.model_validate_json(out_path.read_text(encoding="utf-8"))
        _print_summary(report, out_path)
        sys.exit(0)

    # ── Verify frozen artifacts ───────────────────────────────────────────
    logger.info("[Step 8] Verifying checksums…")
    verify_checksums(args.checksums)
    logger.info("[Step 8] Checksums OK.")

    # ── Check inputs exist ────────────────────────────────────────────────
    for label, fpath in [("eval", args.eval), ("results", args.results)]:
        if not Path(fpath).exists():
            print(f"[FAIL] {label} file not found: {fpath}", file=sys.stderr)
            sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────
    logger.info("[Step 8] Loading eval cases…")
    eval_cases = _load_eval(Path(args.eval))
    logger.info("[Step 8] %d eval cases loaded.", len(eval_cases))

    logger.info("[Step 8] Loading results…")
    results = _load_results(Path(args.results))
    logger.info("[Step 8] %d results loaded.", len(results))

    # ── Compute ───────────────────────────────────────────────────────────
    logger.info("[Step 8] Computing metrics…")
    report = compute_metrics(results, eval_cases)

    # ── Save and freeze ───────────────────────────────────────────────────
    digest = save_metrics(
        report,
        path=out_path,
        checksums_path=Path(args.checksums),
        force=args.force,
    )
    logger.info("[Step 8] metrics_baseline.json frozen.")
    _print_summary(report, out_path, digest)
    print("[Step 8] Baseline metrics complete.")


if __name__ == "__main__":
    main()
