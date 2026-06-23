"""
CLI entrypoint for the deterministic baseline pipeline (Phase 1–3, no LLM).

Usage:
    python -m aml_copilot.step7_runner.run_baseline
    python -m aml_copilot.step7_runner.run_baseline --eval data/fixtures/eval.jsonl \\
        --out artifacts/results.jsonl

Startup (once, not timed per case):
  1. verify_checksums  — abort on any frozen-file mismatch
  2. load_transactions + build graph / overlay / pattern_map  (Step 0 / Step 3)
  3. load_ofac_records + build_ofac_index  (Step 2)
  4. build_feature_matrix + score_accounts → score_map  (Step 5 robust-z)
  5. read eval.jsonl → list[EvalCase]

Per case (90 iterations, each timed with perf_counter):
  screen_account  → list[SanctionsHit]          (Step 2)
  resolve_entity  → EntityChain                  (Step 3)
  build_account_window + evaluate_rules          (Step 4)
  score_map.get   → Optional[AnomalyScore]       (Step 5, O(1))
  apply_decision_table → CaseResult

Output:
  artifacts/results.jsonl  (atomic temp + rename to prevent partial writes)

OFAC name safety:
  SanctionsHit.assigned_name is the synthetic Faker name — not an OFAC
  canonical name.  No OFAC canonical name is serialised into results.jsonl.
  Logging of assigned_name is intentionally absent from this module.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from aml_copilot.schemas import CaseResult, EvalCase
from aml_copilot.step0_scaffold.data_loader import derive_accounts, load_transactions
from aml_copilot.step1_identity.ofac_reader import load_ofac_records
from aml_copilot.step2_sanctions.index import build_ofac_index
from aml_copilot.step2_sanctions.screen import screen_account
from aml_copilot.step3_entity.resolve import (
    build_overlay_map,
    build_pattern_map,
    build_transaction_graph,
    parse_patterns_file,
    resolve_entity,
)
from aml_copilot.step4_rules.engine import build_account_window, evaluate_rules
from aml_copilot.step5_anomaly.features import build_feature_matrix
from aml_copilot.step5_anomaly.scorer import score_accounts
from aml_copilot.step7_runner.decision import apply_decision_table
from aml_copilot.utils.checksum import verify_checksums

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parents[3]

_DEFAULTS: dict[str, Path] = {
    "eval":      _ROOT / "data/fixtures/eval.jsonl",
    "out":       _ROOT / "artifacts/results.jsonl",
    "trans":     _ROOT / "data/raw/HI-Small_Trans.csv",
    "overlay":   _ROOT / "data/processed/identity_overlay.parquet",
    "patterns":  _ROOT / "data/raw/HI-Small_Patterns.txt",
    "ofac_sdn":  _ROOT / "data/raw/ofac/sdn_advanced.xml",
    "ofac_cons": _ROOT / "data/raw/ofac/cons_advanced.xml",
    "checksums": _ROOT / "artifacts/checksums.sha256",
}

EXPECTED_EVAL_SIZE: int = 90


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(EvalCase.model_validate_json(line))
    return cases


def _print_summary(results: list[CaseResult], out_path: Path) -> None:
    labels = Counter(r.disposition for r in results)
    latencies = np.array([r.latency_ms for r in results])
    reasons = Counter(r.decision_reason for r in results)
    print(f"\n{'=' * 64}")
    print(f"results.jsonl: {out_path}  ({len(results)} cases)")
    print(f"  Dispositions:   {dict(sorted(labels.items()))}")
    print(f"  Reasons:        {dict(sorted(reasons.items()))}")
    print(f"  Latency p50:    {np.percentile(latencies, 50):.1f} ms")
    print(f"  Latency p95:    {np.percentile(latencies, 95):.1f} ms")
    print(f"  Latency total:  {latencies.sum():.0f} ms")
    print(f"{'=' * 64}\n")


# ── Public API (callable from tests without subprocess) ──────────────────────

def run(
    eval_path: Path,
    out_path: Path,
    trans_path: Path,
    overlay_path: Path,
    patterns_path: Path,
    ofac_sdn_path: Path,
    ofac_cons_path: Path | None,
    checksums_path: Path,
) -> list[CaseResult]:
    """
    Execute the full baseline pipeline and return list[CaseResult].

    Writes results atomically to out_path.  Raises on any error — no silent
    CLEAR fallback.  Checksums are verified before any processing begins.
    """
    # ── 1. Verify frozen artifacts ────────────────────────────────────────
    logger.info("[Step 7] Verifying checksums…")
    verify_checksums(str(checksums_path))
    logger.info("[Step 7] Checksums OK.")

    # ── 2. Load transactions + graph / overlay / patterns ─────────────────
    logger.info("[Step 7] Loading transactions: %s", trans_path)
    df = load_transactions(str(trans_path))
    logger.info("[Step 7] %d rows loaded.", len(df))

    logger.info("[Step 7] Building transaction graph…")
    adjacency = build_transaction_graph(df)

    logger.info("[Step 7] Loading identity overlay…")
    overlay_map = build_overlay_map(str(overlay_path))

    logger.info("[Step 7] Parsing patterns file…")
    patterns = parse_patterns_file(str(patterns_path))
    pattern_map = build_pattern_map(patterns)

    # ── 3. Build OFAC index ───────────────────────────────────────────────
    logger.info("[Step 7] Loading OFAC records…")
    ofac_records = load_ofac_records(
        sdn_path=ofac_sdn_path,
        cons_path=ofac_cons_path,
    )
    logger.info("[Step 7] %d OFAC records loaded.", len(ofac_records))
    ofac_index = build_ofac_index(ofac_records)
    logger.info("[Step 7] OFAC index: %d entries.", len(ofac_index.all_entries))

    # ── 4. Fit anomaly model + pre-score all accounts ─────────────────────
    logger.info("[Step 7] Building feature matrix…")
    accounts_df = derive_accounts(df)
    feat_df = build_feature_matrix(df, accounts_df)

    logger.info("[Step 7] Computing robust-z anomaly scores…")
    all_scores = score_accounts(feat_df)
    score_map = {s.account_id: s for s in all_scores}
    n_flagged = sum(1 for s in all_scores if s.is_flagged)
    logger.info("[Step 7] %d / %d accounts flagged.", n_flagged, len(score_map))

    # ── 5. Load eval cases ────────────────────────────────────────────────
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Eval set not found: {eval_path}\n"
            "Run python -m aml_copilot.step6_eval.builder first."
        )
    cases = _load_eval_cases(eval_path)
    if len(cases) != EXPECTED_EVAL_SIZE:
        raise RuntimeError(
            f"Expected {EXPECTED_EVAL_SIZE} eval cases, got {len(cases)} — "
            f"eval.jsonl may be truncated or rebuilt"
        )
    logger.info("[Step 7] %d eval cases loaded.", len(cases))

    # ── 6. Per-case processing loop ───────────────────────────────────────
    results: list[CaseResult] = []

    for i, case in enumerate(cases, start=1):
        if i % 10 == 0 or i == 1:
            logger.info("  Processing case %d / %d…", i, len(cases))

        try:
            t0 = time.perf_counter()

            # Step 2: sanctions screen — uses assigned Faker name, not OFAC canonical name
            profile = overlay_map.get(case.account_id)
            if profile is None:
                raise RuntimeError(
                    f"Account {case.account_id} absent from identity overlay — "
                    "this indicates an eval-set construction error"
                )
            assigned_name: str = profile["name"]
            hits = screen_account(case.account_id, assigned_name, ofac_index)

            # Step 3: entity resolve
            entity = resolve_entity(
                case.account_id, adjacency, overlay_map, pattern_map
            )

            # Step 4: rule evaluation
            window = build_account_window(df, case.account_id)
            firings = evaluate_rules(case.account_id, window, entity)

            # Step 5: anomaly lookup — O(1); None if account absent from score_map
            anomaly = score_map.get(case.account_id)

            latency_ms = (time.perf_counter() - t0) * 1000

        except Exception as exc:
            raise RuntimeError(
                f"Case {case.case_id!r} (account {case.account_id!r}) failed "
                f"during tool execution: {exc}"
            ) from exc

        result = apply_decision_table(
            case_id=case.case_id,
            account_id=case.account_id,
            sanctions_hits=hits,
            rule_firings=firings,
            anomaly_score=anomaly,
            latency_ms=latency_ms,
        )
        results.append(result)

    # ── 7. Post-loop validation ───────────────────────────────────────────
    if len(results) != len(cases):
        raise RuntimeError(
            f"Result count mismatch: produced {len(results)}, expected {len(cases)}"
        )
    dup_case_ids = [cid for cid, n in Counter(r.case_id for r in results).items() if n > 1]
    if dup_case_ids:
        raise RuntimeError(f"Duplicate case_ids in results: {dup_case_ids}")

    # ── 8. Atomic write ───────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    content = "\n".join(r.model_dump_json() for r in results) + "\n"
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(out_path)
    logger.info("[Step 7] results.jsonl written: %s", out_path)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AML copilot deterministic baseline runner (Phase 1–3, no LLM)"
    )
    p.add_argument("--eval",      default=str(_DEFAULTS["eval"]),
                   help="Path to frozen eval.jsonl")
    p.add_argument("--out",       default=str(_DEFAULTS["out"]),
                   help="Output path for results.jsonl")
    p.add_argument("--trans",     default=str(_DEFAULTS["trans"]),
                   help="HI-Small_Trans.csv path")
    p.add_argument("--overlay",   default=str(_DEFAULTS["overlay"]),
                   help="identity_overlay.parquet path")
    p.add_argument("--patterns",  default=str(_DEFAULTS["patterns"]),
                   help="HI-Small_Patterns.txt path")
    p.add_argument("--ofac-sdn",  default=str(_DEFAULTS["ofac_sdn"]),
                   dest="ofac_sdn", help="sdn_advanced.xml path")
    p.add_argument("--ofac-cons", default=str(_DEFAULTS["ofac_cons"]),
                   dest="ofac_cons", help="cons_advanced.xml path (optional)")
    p.add_argument("--checksums", default=str(_DEFAULTS["checksums"]),
                   help="artifacts/checksums.sha256 path")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    try:
        results = run(
            eval_path=Path(args.eval),
            out_path=Path(args.out),
            trans_path=Path(args.trans),
            overlay_path=Path(args.overlay),
            patterns_path=Path(args.patterns),
            ofac_sdn_path=Path(args.ofac_sdn),
            ofac_cons_path=Path(args.ofac_cons) if args.ofac_cons else None,
            checksums_path=Path(args.checksums),
        )
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)

    _print_summary(results, Path(args.out))
    print("[Step 7] Baseline run complete.")


if __name__ == "__main__":
    main()
