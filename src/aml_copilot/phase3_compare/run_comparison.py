"""
Phase 3 M5 — cross-framework comparison runner.

Executes all registered AMLAgentRunner implementations independently,
computes per-framework and cross-framework metrics, verifies agreement,
and writes phase3_comparison_metrics.json.

Adding Framework #4
-------------------
Register one new entry in RUNNER_REGISTRY at the bottom of the imports section.
No other code needs to change.

CLI
---
    python -m aml_copilot.phase3_compare.run_comparison \\
        --eval   data/fixtures/eval.jsonl \\
        --baseline artifacts/results.jsonl \\
        --out    artifacts/phase3_comparison_metrics.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from aml_copilot.phase3_compare._shared import EXPECTED_EVAL_SIZE, load_eval_cases
from aml_copilot.phase3_compare.crewai_runner import CrewAIRunner
from aml_copilot.phase3_compare.langgraph_runner import LangGraphRunner
from aml_copilot.phase3_compare.metrics import (
    compute_comparison_metrics,
    compute_framework_metrics,
    get_framework_versions,
)
from aml_copilot.phase3_compare.openai_agents_runner import OpenAIAgentsRunner
from aml_copilot.schemas import EvalCase, Phase3CaseResult, Phase3ComparisonMetrics

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parents[3]           # project root
_PKG  = Path(__file__).parent               # phase3_compare/
_P2   = Path(__file__).parents[1] / "phase2_eval"

_CANONICAL_EVAL: Path = _ROOT / "data" / "fixtures" / "eval.jsonl"

_DEFAULTS: dict[str, Path] = {
    "eval":      _ROOT / "data/fixtures/eval.jsonl",
    "baseline":  _ROOT / "artifacts/results.jsonl",
    "out":       _ROOT / "artifacts/phase3_comparison_metrics.json",
    "p1_metrics": _ROOT / "artifacts/metrics_baseline.json",
    "p2_metrics": _ROOT / "artifacts/phase2_langgraph_metrics.json",
}

# ── Runner registry ───────────────────────────────────────────────────────────
# Single registration point.  To add Framework #4, append one tuple here:
#   (MyNewRunner, _PKG / "my_new_runner.py")
# Framework execution order is the order of this list.

RUNNER_REGISTRY: list[tuple[Any, Path]] = [
    (LangGraphRunner,    _PKG / "langgraph_runner.py"),
    (CrewAIRunner,       _PKG / "crewai_runner.py"),
    (OpenAIAgentsRunner, _PKG / "openai_agents_runner.py"),
]


# ── Canonical eval-path safeguard ─────────────────────────────────────────────


def _is_canonical_eval_path(path: Path) -> bool:
    """Return True when *path* resolves to the committed canonical eval fixture."""
    try:
        return path.resolve() == _CANONICAL_EVAL.resolve()
    except OSError:
        return False


def _validate_eval_mode(eval_path: Path, cases: list[EvalCase]) -> None:
    """Require exactly EXPECTED_EVAL_SIZE cases when the canonical eval path is used.

    Mini fixtures and ad-hoc paths bypass this check; only the frozen production
    fixture at data/fixtures/eval.jsonl enforces the 90-case requirement.
    """
    if _is_canonical_eval_path(eval_path) and len(cases) != EXPECTED_EVAL_SIZE:
        raise RuntimeError(
            f"Canonical Phase 3 evaluation requires {EXPECTED_EVAL_SIZE} cases; "
            f"got {len(cases)} in {eval_path}"
        )


# ── Core runner ───────────────────────────────────────────────────────────────


def run(
    eval_path: Path,
    baseline_path: Path,
    out_path: Path,
    *,
    p1_metrics_path: Path | None = None,
    p2_metrics_path: Path | None = None,
) -> Phase3ComparisonMetrics:
    """Execute the comparison and write the output artifact.

    Parameters
    ----------
    eval_path : Path
        data/fixtures/eval.jsonl — 90 frozen EvalCase rows.
    baseline_path : Path
        artifacts/results.jsonl — 90 Phase 1 CaseResult rows.
    out_path : Path
        Destination for phase3_comparison_metrics.json (not frozen).
    p1_metrics_path : Path, optional
        Phase 1 metrics JSON; default artifacts/metrics_baseline.json.
    p2_metrics_path : Path, optional
        Phase 2 metrics JSON; default artifacts/phase2_langgraph_metrics.json.

    Returns
    -------
    Phase3ComparisonMetrics

    Raises
    ------
    FileNotFoundError
        If eval_path or baseline_path do not exist.
    """
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval set not found: {eval_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline results not found: {baseline_path}\n"
            "Run python -m aml_copilot.step7_runner.run_baseline first."
        )

    eval_cases = load_eval_cases(eval_path)
    _validate_eval_mode(eval_path, eval_cases)

    p1_acc = _load_accuracy(p1_metrics_path or _DEFAULTS["p1_metrics"], "disposition_accuracy")
    p2_acc = _load_accuracy(p2_metrics_path or _DEFAULTS["p2_metrics"], "disposition_accuracy")
    framework_versions = get_framework_versions()

    # ── Execute each runner independently ─────────────────────────────────────
    framework_results: dict[str, list[Phase3CaseResult]] = {}
    runner_errors: dict[str, Exception] = {}

    for runner_cls, runner_file in RUNNER_REGISTRY:
        fw = runner_cls.framework_name
        logger.info("[M5] Running %s …", fw)
        t0 = time.perf_counter()
        runner = runner_cls()
        try:
            results = runner.run(eval_path, baseline_path)
            framework_results[fw] = results
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("[M5] %s done in %.0f ms  (%d results)", fw, elapsed, len(results))
        except Exception as exc:
            runner_errors[fw] = exc
            logger.error("[M5] %s FAILED: %s", fw, exc)

    # ── Compute per-framework metrics (registry order preserved) ──────────────
    framework_metrics = []
    for runner_cls, runner_file in RUNNER_REGISTRY:
        fw = runner_cls.framework_name
        if fw in framework_results:
            m = compute_framework_metrics(
                framework_results[fw], eval_cases, runner_file
            )
            framework_metrics.append(m)
            logger.info(
                "[M5] %s  accuracy=%.4f  FCR=%.4f  LOC=%d",
                fw, m.disposition_accuracy, m.false_clear_rate_weighted, m.loc,
            )

    # ── Build comparison report ────────────────────────────────────────────────
    comparison = compute_comparison_metrics(
        framework_results=framework_results,
        framework_metrics=framework_metrics,
        eval_cases=eval_cases,
        phase1_accuracy=p1_acc,
        phase2_accuracy=p2_acc,
        framework_version_information=framework_versions,
        runner_errors=runner_errors,
    )

    # ── Write output artifact (atomic; not frozen) ─────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(out_path)
    logger.info("[M5] Artifact written: %s", out_path)

    return comparison


# ── Formatting helpers ────────────────────────────────────────────────────────


def _format_table(comparison: Phase3ComparisonMetrics) -> str:
    """Render a fixed-width table of per-framework metrics."""
    col_w = {
        "framework":  15,
        "accuracy":    8,
        "fcr":         8,
        "override":    9,
        "humrev":      9,
        "p50":         8,
        "p95":         8,
        "avg":         8,
        "loc":         5,
    }
    header = (
        f"{'Framework':<{col_w['framework']}}"
        f"{'Accuracy':>{col_w['accuracy']}}"
        f"{'FCR(wt)':>{col_w['fcr']}}"
        f"{'Override':>{col_w['override']}}"
        f"{'HumRev':>{col_w['humrev']}}"
        f"{'p50ms':>{col_w['p50']}}"
        f"{'p95ms':>{col_w['p95']}}"
        f"{'avgms':>{col_w['avg']}}"
        f"{'LOC':>{col_w['loc']}}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for m in comparison.frameworks:
        row = (
            f"{m.framework:<{col_w['framework']}}"
            f"{m.disposition_accuracy:>{col_w['accuracy']}.4f}"
            f"{m.false_clear_rate_weighted:>{col_w['fcr']}.4f}"
            f"{m.override_rate:>{col_w['override']}.2%}"
            f"{m.human_review_rate:>{col_w['humrev']}.2%}"
            f"{m.latency_p50_ms:>{col_w['p50']}.2f}"
            f"{m.latency_p95_ms:>{col_w['p95']}.2f}"
            f"{m.average_latency_ms:>{col_w['avg']}.2f}"
            f"{m.loc:>{col_w['loc']}}"
        )
        rows.append(row)
    return "\n".join(rows)


def _yn(value: bool, extra: str = "") -> str:
    tag = "YES" if value else "NO "
    return f"{tag}  {extra}".rstrip()


def print_comparison(
    comparison: Phase3ComparisonMetrics,
    runner_errors: dict[str, Exception],
    out_path: Path,
    elapsed_s: float,
) -> None:
    """Print the formatted comparison report to stdout."""
    bar = "=" * 60

    print(f"\n{bar}")
    print("Phase 3 Framework Comparison")
    print(f"{bar}\n")

    n = comparison.eval_size
    print(f"Eval size : {n} cases")
    print(f"Frameworks: {len(comparison.frameworks)} executed  "
          f"({len(runner_errors)} failed)")
    print()

    print("Framework Metrics")
    print("-" * 40)
    print(_format_table(comparison))
    print()

    print("Agreement Summary")
    print("-" * 40)

    if len(comparison.frameworks) >= 2:
        n_cases = comparison.eval_size
        print(f"  Dispositions agree  : {_yn(comparison.all_dispositions_agree, f'({n_cases}/{n_cases})')}")
        print(f"  Reasoning agrees    : {_yn(comparison.all_reasoning_agree,   f'({n_cases}/{n_cases})')}")
        print(f"  Human-review flags  : {_yn(comparison.all_human_review_flags_agree)}")
        print(f"  All costs zero      : {_yn(comparison.all_costs_zero)}")
        print(f"  All tokens zero     : {_yn(comparison.all_tokens_zero)}")
    else:
        print("  (insufficient frameworks for agreement check)")
    print()

    if runner_errors:
        print("Failed Frameworks")
        print("-" * 40)
        for fw, exc in runner_errors.items():
            print(f"  {fw}: {type(exc).__name__}: {exc}")
        print()

    print("Accuracy Reference")
    print("-" * 40)
    print(f"  Phase 1 baseline : {comparison.phase1_accuracy:.6f}")
    print(f"  Phase 2 LangGraph: {comparison.phase2_accuracy:.6f}")
    if comparison.frameworks:
        accs = {m.framework: m.disposition_accuracy for m in comparison.frameworks}
        for fw, acc in accs.items():
            print(f"  Phase 3 {fw:<13}: {acc:.6f}")
    print()

    fw_vers = comparison.framework_version_information
    if fw_vers:
        print("Framework Versions")
        print("-" * 40)
        for k, v in sorted(fw_vers.items()):
            print(f"  {k:<20}: {v}")
        print()

    print(bar)
    verdict_line = "PASS — All frameworks produce identical results." \
        if comparison.comparison_passed \
        else "FAIL — Framework disagreement or runner error detected."
    print(f"VERDICT: {'PASS' if comparison.comparison_passed else 'FAIL'}")
    print(verdict_line)
    print(bar)
    print()
    print(f"Artifact : {out_path}")
    print(f"Elapsed  : {elapsed_s:.2f} s")
    print()


# ── I/O helpers ───────────────────────────────────────────────────────────────


def _load_accuracy(path: Path, key: str) -> float:
    """Read a float value from a JSON metrics file; return 0.0 if file not found."""
    if not path.exists():
        logger.warning("Metrics file not found: %s — using 0.0 for %s", path, key)
        return 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data[key])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s from %s: %s — using 0.0", key, path, exc)
        return 0.0


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3 cross-framework AML comparison runner (M5)"
    )
    p.add_argument("--eval",      default=str(_DEFAULTS["eval"]))
    p.add_argument("--baseline",  default=str(_DEFAULTS["baseline"]))
    p.add_argument("--out",       default=str(_DEFAULTS["out"]))
    p.add_argument(
        "--p1-metrics", default=str(_DEFAULTS["p1_metrics"]),
        help="Path to Phase 1 metrics_baseline.json",
    )
    p.add_argument(
        "--p2-metrics", default=str(_DEFAULTS["p2_metrics"]),
        help="Path to Phase 2 phase2_langgraph_metrics.json",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    eval_path     = Path(args.eval)
    baseline_path = Path(args.baseline)
    out_path      = Path(args.out)
    p1_path       = Path(args.p1_metrics)
    p2_path       = Path(args.p2_metrics)

    # Pre-flight checks
    for label, fpath in [("eval", eval_path), ("baseline", baseline_path)]:
        if not fpath.exists():
            print(f"[FAIL] {label} file not found: {fpath}", file=sys.stderr)
            sys.exit(1)

    t_start = time.perf_counter()

    # Track runner errors separately for the summary (run() handles isolation)
    runner_errors: dict[str, Exception] = {}
    _orig_run_one = _run_one_isolated

    comparison = run(
        eval_path=eval_path,
        baseline_path=baseline_path,
        out_path=out_path,
        p1_metrics_path=p1_path,
        p2_metrics_path=p2_path,
    )

    # Reconstruct runner_errors from comparison (frameworks in registry not in comparison)
    executed_fws = {m.framework for m in comparison.frameworks}
    registry_fws = {cls.framework_name for cls, _ in RUNNER_REGISTRY}
    failed_fws = registry_fws - executed_fws
    runner_errors = {fw: RuntimeError("Execution failed") for fw in failed_fws}

    elapsed_s = time.perf_counter() - t_start

    print_comparison(comparison, runner_errors, out_path, elapsed_s)

    sys.exit(0 if comparison.comparison_passed else 1)


def _run_one_isolated(
    runner_cls: Any,
    runner_file: Path,
    eval_path: Path,
    baseline_path: Path,
) -> tuple[str, list[Phase3CaseResult] | None, Exception | None]:
    """Execute one runner; return (fw_name, results_or_None, error_or_None)."""
    fw = runner_cls.framework_name
    runner = runner_cls()
    try:
        results = runner.run(eval_path, baseline_path)
        return (fw, results, None)
    except Exception as exc:
        return (fw, None, exc)


if __name__ == "__main__":
    main()
