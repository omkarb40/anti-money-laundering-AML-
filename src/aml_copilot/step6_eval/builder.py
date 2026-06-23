"""Step 6 — Eval set builder.

Assembles data/fixtures/eval.jsonl with exactly 90 EvalCase objects from 5 slices.
Gold labels come from IBM ground truth (ibm_labeled, typology, sanctions_hit) or the
inline decision table (conflict cases, near-miss).  Never from Step 7 output.

Run:
    python -m aml_copilot.step6_eval.builder
    python -m aml_copilot.step6_eval.builder --force   # rebuild if already frozen
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import polars as pl

from aml_copilot.schemas import AnomalyScore, EvalCase, RuleFiring
from aml_copilot.step0_scaffold.data_loader import derive_accounts, load_transactions
from aml_copilot.step3_entity.resolve import parse_patterns_file
from aml_copilot.step4_rules.engine import build_account_window, evaluate_rules
from aml_copilot.step5_anomaly.features import build_feature_matrix
from aml_copilot.step5_anomaly.scorer import fit_model, score_accounts
from aml_copilot.utils.checksum import append_checksum, compute_sha256, verify_checksums

logger = logging.getLogger(__name__)

# ── Default paths (relative to project root) ─────────────────────────────────

_ROOT = Path(__file__).parents[3]
_TRANS_PATH    = _ROOT / "data/raw/HI-Small_Trans.csv"
_OVERLAY_PATH  = _ROOT / "data/processed/identity_overlay.parquet"
_GT_PATH       = _ROOT / "data/fixtures/ground_truth_matches.csv"
_PATTERNS_PATH = _ROOT / "data/raw/HI-Small_Patterns.txt"
_EVAL_OUT      = _ROOT / "data/fixtures/eval.jsonl"
_CHECKSUM_FILE = _ROOT / "artifacts/checksums.sha256"

# ── Constants ─────────────────────────────────────────────────────────────────

EVAL_SEED: int = 42
EVAL_SIZE: int = 90

SLICE_COUNTS: dict[str, int] = {
    "ibm_labeled":          30,
    "sanctions_hit":        15,
    "sanctions_near_miss":  15,
    "rules_anomaly_conflict": 10,
    "typology":             20,
}

# Typologies from HI-Small_Patterns.txt, normalised from "FAN-OUT" → "fan_out"
TYPOLOGY_TARGETS: dict[str, int] = {
    "fan_out":       3,
    "fan_in":        3,
    "cycle":         3,
    "scatter_gather": 3,
    "bipartite":     2,
    "stack":         2,
    "gather_scatter": 2,
    "random":        2,
}

CONFLICT_TARGETS: dict[str, int] = {
    "anomaly_no_rule":  4,
    "rule_no_anomaly":  4,
    "rule3_no_anomaly": 2,
}

# Non-flagged accounts to randomly sample for conflict rule/anomaly pools
_NON_FLAGGED_SAMPLE: int = 3000


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalize_typology(raw: str) -> str:
    """'FAN-OUT' → 'fan_out', 'SCATTER-GATHER' → 'scatter_gather', etc."""
    return raw.strip().upper().replace("-", "_").replace(" ", "_").lower()


def _max_sev(firings: list[RuleFiring]) -> int:
    return max((f.severity for f in firings), default=0)


def _decision_gold(
    has_sanctions_90: bool,
    firings: list[RuleFiring],
    anomaly: Optional[AnomalyScore],
) -> str:
    """
    Inline decision table — mirrors Step 7 without importing it.

    Branch 1: sanctions >= 0.90 OR severity-3 rule  → ESCALATE
    Branch 2: is_flagged AND severity-2+ rule        → ESCALATE
    Branch 3: (everything else)                      → CLEAR
    """
    is_flagged = anomaly.is_flagged if anomaly is not None else False
    max_sv = _max_sev(firings)
    if has_sanctions_90 or max_sv >= 3:
        return "ESCALATE"
    if is_flagged and max_sv >= 2:
        return "ESCALATE"
    return "CLEAR"


def _txn_ids(df_idx: pl.DataFrame, account_id: str, n: int = 5) -> list[str]:
    """Return string row-indices of the most recent n transactions for account_id."""
    rows = df_idx.filter(
        (pl.col("from_account") == account_id)
        | (pl.col("to_account") == account_id)
    )
    if len(rows) == 0:
        return []
    return rows["_row_nr"].tail(n).cast(pl.String).to_list()


def _batch_step4(
    df: pl.DataFrame,
    account_ids: list[str],
    tag: str,
) -> dict[str, list[RuleFiring]]:
    """
    Run Step 4 (no entity / no CORRIDOR_001) on a batch of accounts.
    Logs progress every 500 accounts.
    """
    out: dict[str, list[RuleFiring]] = {}
    n = len(account_ids)
    for i, aid in enumerate(account_ids):
        if i % 500 == 0 and i > 0:
            logger.info("  %s step4: %d / %d", tag, i, n)
        out[aid] = evaluate_rules(aid, build_account_window(df, aid), entity=None)
    logger.info("  %s step4: %d / %d done", tag, n, n)
    return out


def _remove_checksum_line(checksum_file: Path, target: str) -> None:
    """Remove the line for *target* from checksum_file (used by --force)."""
    if not checksum_file.exists():
        return
    lines = checksum_file.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = [ln for ln in lines if target not in ln]
    checksum_file.write_text("".join(new_lines), encoding="utf-8")


# ── Slice builders ────────────────────────────────────────────────────────────

def _build_sanctions_hit(
    gt_rows: list[dict],
    df_idx: pl.DataFrame,
    used: set[str],
) -> list[EvalCase]:
    """
    15 sanctions-hit cases: TPs with expected_score_min >= 0.90.
    Gold = ESCALATE (Branch 1 guaranteed: score >= 0.90 → sanctions trigger).
    Flavors exact(5) + transliteration(5) + partial_reorder(5) = 15.
    """
    eligible = [
        r for r in gt_rows
        if r["gold_is_match"] == "True"
        and float(r["expected_score_min"]) >= 0.90
        and r["account_id"] not in used
    ]
    need = SLICE_COUNTS["sanctions_hit"]
    if len(eligible) < need:
        raise RuntimeError(
            f"sanctions_hit: need {need} TPs with score_min>=0.90, found {len(eligible)}"
        )
    cases: list[EvalCase] = []
    for row in eligible[:need]:
        aid = row["account_id"]
        cases.append(EvalCase(
            case_id=f"SH_{aid}",
            account_id=aid,
            gold_label="ESCALATE",
            case_type="sanctions_hit",
            relevant_txn_ids=_txn_ids(df_idx, aid),
            notes=f"TP {row['match_flavor']}; expected_score_min={row['expected_score_min']}",
        ))
        used.add(aid)
    return cases


def _build_near_miss(
    gt_rows: list[dict],
    df_idx: pl.DataFrame,
    hn_firings: dict[str, list[RuleFiring]],
    score_map: dict[str, AnomalyScore],
    used: set[str],
    rng: random.Random,
) -> list[EvalCase]:
    """
    15 near-miss cases: HNs (gold_is_match=False) that pass pre-screen.
    Pre-screen: discard any HN where the inline decision table would ESCALATE
    (ensures gold = CLEAR is consistent with the decision table).
    """
    hn_rows = [
        r for r in gt_rows
        if r["gold_is_match"] == "False"
        and r["account_id"] not in used
    ]
    clear_hns = [
        r for r in hn_rows
        if _decision_gold(
            False,
            hn_firings.get(r["account_id"], []),
            score_map.get(r["account_id"]),
        ) == "CLEAR"
    ]
    need = SLICE_COUNTS["sanctions_near_miss"]
    if len(clear_hns) < need:
        raise RuntimeError(
            f"sanctions_near_miss: need {need} CLEAR HNs after pre-screen, "
            f"found {len(clear_hns)} (total HNs available: {len(hn_rows)})"
        )
    rng.shuffle(clear_hns)
    cases: list[EvalCase] = []
    for row in clear_hns[:need]:
        aid = row["account_id"]
        cases.append(EvalCase(
            case_id=f"SNM_{aid}",
            account_id=aid,
            gold_label="CLEAR",
            case_type="sanctions_near_miss",
            relevant_txn_ids=_txn_ids(df_idx, aid),
            notes=(
                f"HN: assigned name shares surname with OFAC target; "
                f"JW {row['expected_score_min']}–{row['expected_score_max']} "
                f"(below 0.90 escalation threshold)"
            ),
        ))
        used.add(aid)
    return cases


def _build_ibm_labeled(
    df_idx: pl.DataFrame,
    ibm_list: list[str],
    ibm_firings: dict[str, list[RuleFiring]],
    used: set[str],
    rng: random.Random,
) -> list[EvalCase]:
    """
    30 IBM-labeled cases: 10 per severity band.
    Gold = ESCALATE (IBM ground truth for every account here).

    Band 3: at least one severity-3 rule fires (STRUCT_001 / PASSTHROUGH_001).
    Band 2: at least one severity-2 rule fires, no severity-3
            (FAN_OUT_001 / FAN_IN_001).
    Band 1: no severity-2 or -3 rule fires (CORRIDOR skipped; entity=None).
    """
    band3 = [a for a in ibm_list if _max_sev(ibm_firings.get(a, [])) >= 3 and a not in used]
    band2 = [a for a in ibm_list if _max_sev(ibm_firings.get(a, [])) == 2 and a not in used]
    band1 = [a for a in ibm_list if _max_sev(ibm_firings.get(a, [])) <= 1 and a not in used]

    per_band = 10
    for name, pool in [("band3", band3), ("band2", band2), ("band1", band1)]:
        if len(pool) < per_band:
            raise RuntimeError(
                f"ibm_labeled {name}: need {per_band} accounts, found {len(pool)}"
            )

    rng.shuffle(band3); rng.shuffle(band2); rng.shuffle(band1)

    cases: list[EvalCase] = []
    for sev_val, accounts in [(3, band3[:per_band]), (2, band2[:per_band]), (1, band1[:per_band])]:
        for aid in accounts:
            firings = ibm_firings.get(aid, [])
            top_rule = (
                max(firings, key=lambda f: f.severity).rule_id
                if firings else "none"
            )
            cases.append(EvalCase(
                case_id=f"IBM_{aid}",
                account_id=aid,
                gold_label="ESCALATE",
                case_type="ibm_labeled",
                severity_band=sev_val,
                relevant_txn_ids=_txn_ids(df_idx, aid),
                notes=f"IBM is_laundering=1; severity_band={sev_val}; top_rule={top_rule}",
            ))
            used.add(aid)
    return cases


def _build_typology(
    df_idx: pl.DataFrame,
    patterns: dict[str, list[str]],
    launder_senders: set[str],
    used: set[str],
    rng: random.Random,
) -> list[EvalCase]:
    """
    20 typology cases from HI-Small_Patterns.txt.
    Gold = ESCALATE (IBM laundering ground truth).

    Pools are filtered to accounts that:
      - appear in HI-Small_Patterns.txt for the target typology
      - also appear as from_account in at least one is_laundering==1 row
      - are not already used in another slice
    Aborts if any typology bucket cannot reach its required count.
    """
    # Normalise typology keys and collect pools
    norm_patterns: dict[str, list[str]] = {}
    for raw_typ, aids in patterns.items():
        norm = _normalize_typology(raw_typ)
        norm_patterns.setdefault(norm, []).extend(aids)

    # Deduplicate within each normalised bucket
    norm_patterns = {
        k: list(dict.fromkeys(v))  # preserves order, drops dupes
        for k, v in norm_patterns.items()
    }

    cases: list[EvalCase] = []
    for typo, count in TYPOLOGY_TARGETS.items():
        pool = [
            a for a in norm_patterns.get(typo, [])
            if a in launder_senders and a not in used
        ]
        if len(pool) < count:
            raise RuntimeError(
                f"typology {typo!r}: need {count} IBM-labeled pattern accounts, "
                f"found {len(pool)} (total in pattern file: {len(norm_patterns.get(typo, []))})"
            )
        rng.shuffle(pool)
        for aid in pool[:count]:
            cases.append(EvalCase(
                case_id=f"TYP_{aid}",
                account_id=aid,
                gold_label="ESCALATE",
                case_type="typology",
                typology=typo,
                relevant_txn_ids=_txn_ids(df_idx, aid),
                notes=f"IBM typology={typo}; appears in HI-Small_Patterns.txt",
            ))
            used.add(aid)
    return cases


def _build_conflict(
    df_idx: pl.DataFrame,
    flagged_firings: dict[str, list[RuleFiring]],
    sample_firings: dict[str, list[RuleFiring]],
    ibm_firings: dict[str, list[RuleFiring]],
    score_map: dict[str, AnomalyScore],
    used: set[str],
    rng: random.Random,
) -> list[EvalCase]:
    """
    10 conflict cases:
      anomaly_no_rule (4): is_flagged=True, no severity>=2 rule → CLEAR (Branch 3)
      rule_no_anomaly (4): severity-2 rule fires, not is_flagged → CLEAR (Branch 3)
      rule3_no_anomaly (2): severity-3 rule fires, not is_flagged → ESCALATE (Branch 1)

    rule3_no_anomaly tries non-flagged sample first; falls back to IBM accounts not yet used.
    """
    # anomaly_no_rule pool
    anr_pool = [
        aid for aid, firings in flagged_firings.items()
        if score_map.get(aid) and score_map[aid].is_flagged
        and _max_sev(firings) < 2
        and aid not in used
    ]

    # rule_no_anomaly pool: sev==2 exactly, not flagged
    rna_pool = [
        aid for aid, firings in sample_firings.items()
        if _max_sev(firings) == 2
        and score_map.get(aid) and not score_map[aid].is_flagged
        and aid not in used
    ]

    # rule3_no_anomaly pool: sev>=3, not flagged (sample first, then IBM fallback)
    r3na_pool = [
        aid for aid, firings in sample_firings.items()
        if _max_sev(firings) >= 3
        and score_map.get(aid) and not score_map[aid].is_flagged
        and aid not in used
    ]
    if len(r3na_pool) < CONFLICT_TARGETS["rule3_no_anomaly"]:
        # Fallback: IBM accounts with sev>=3 rule, not flagged, not used
        for aid, firings in ibm_firings.items():
            if (
                _max_sev(firings) >= 3
                and aid not in used
                and score_map.get(aid) and not score_map[aid].is_flagged
                and aid not in r3na_pool
            ):
                r3na_pool.append(aid)

    for name, pool, need in [
        ("anomaly_no_rule",  anr_pool,  CONFLICT_TARGETS["anomaly_no_rule"]),
        ("rule_no_anomaly",  rna_pool,  CONFLICT_TARGETS["rule_no_anomaly"]),
        ("rule3_no_anomaly", r3na_pool, CONFLICT_TARGETS["rule3_no_anomaly"]),
    ]:
        if len(pool) < need:
            raise RuntimeError(
                f"conflict {name}: need {need}, found {len(pool)}"
            )

    rng.shuffle(anr_pool); rng.shuffle(rna_pool); rng.shuffle(r3na_pool)

    cases: list[EvalCase] = []

    for aid in anr_pool[: CONFLICT_TARGETS["anomaly_no_rule"]]:
        pct = score_map[aid].percentile
        cases.append(EvalCase(
            case_id=f"CONF_ANR_{aid}",
            account_id=aid,
            gold_label="CLEAR",
            case_type="rules_anomaly_conflict",
            conflict_type="anomaly_no_rule",
            relevant_txn_ids=_txn_ids(df_idx, aid),
            notes=f"anomaly flagged (pct={pct:.4f}) but no sev>=2 rule; decision table → CLEAR",
        ))
        used.add(aid)

    for aid in rna_pool[: CONFLICT_TARGETS["rule_no_anomaly"]]:
        firings = sample_firings[aid]
        top = max(firings, key=lambda f: f.severity).rule_id
        cases.append(EvalCase(
            case_id=f"CONF_RNA_{aid}",
            account_id=aid,
            gold_label="CLEAR",
            case_type="rules_anomaly_conflict",
            conflict_type="rule_no_anomaly",
            relevant_txn_ids=_txn_ids(df_idx, aid),
            notes=f"rule {top} (sev=2) fires but anomaly not flagged; decision table → CLEAR",
        ))
        used.add(aid)

    for aid in r3na_pool[: CONFLICT_TARGETS["rule3_no_anomaly"]]:
        # firings may come from sample or IBM fallback
        firings = sample_firings.get(aid) or ibm_firings.get(aid, [])
        top = max(firings, key=lambda f: f.severity).rule_id if firings else "unknown"
        cases.append(EvalCase(
            case_id=f"CONF_R3NA_{aid}",
            account_id=aid,
            gold_label="ESCALATE",
            case_type="rules_anomaly_conflict",
            conflict_type="rule3_no_anomaly",
            relevant_txn_ids=_txn_ids(df_idx, aid),
            notes=f"rule {top} (sev=3) fires without anomaly flagging; decision table → ESCALATE",
        ))
        used.add(aid)

    return cases


# ── Public API ────────────────────────────────────────────────────────────────

def build_eval_set(
    trans_path: str | Path = _TRANS_PATH,
    overlay_path: str | Path = _OVERLAY_PATH,
    ground_truth_path: str | Path = _GT_PATH,
    patterns_path: str | Path = _PATTERNS_PATH,
) -> list[EvalCase]:
    """
    Build exactly 90 EvalCase objects across 5 slices.
    Raises RuntimeError if any slice cannot reach its required count.
    """
    rng = random.Random(EVAL_SEED)

    for p in [trans_path, overlay_path, ground_truth_path, patterns_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("[Step 6] Loading transactions (%s)…", trans_path)
    df = load_transactions(str(trans_path))
    df_idx = df.with_row_index("_row_nr")
    logger.info("[Step 6] %d transactions loaded.", len(df))

    launder_senders: set[str] = set(
        df.filter(pl.col("is_laundering") == 1)["from_account"].to_list()
    )
    logger.info("[Step 6] %d unique IBM-labeled sender accounts.", len(launder_senders))

    logger.info("[Step 6] Loading identity overlay…")
    overlay_df = pl.read_parquet(str(overlay_path))
    overlay_ids: set[str] = set(overlay_df["account_id"].to_list())

    logger.info("[Step 6] Parsing ground truth CSV…")
    with open(ground_truth_path, newline="", encoding="utf-8") as fh:
        gt_rows = list(csv.DictReader(fh))

    logger.info("[Step 6] Parsing HI-Small_Patterns.txt…")
    patterns = parse_patterns_file(str(patterns_path))
    logger.info("[Step 6] %d typologies in pattern file.", len(patterns))

    # ── Step 5: fit IsolationForest on full account population ────────────────
    logger.info("[Step 6] Building feature matrix + fitting IsolationForest…")
    accounts_df = derive_accounts(df)
    feat_df = build_feature_matrix(df, accounts_df)
    model = fit_model(feat_df)
    all_scores = score_accounts(feat_df, model)
    score_map: dict[str, AnomalyScore] = {s.account_id: s for s in all_scores}
    n_flagged = sum(1 for s in all_scores if s.is_flagged)
    logger.info("[Step 6] %d / %d accounts flagged.", n_flagged, len(all_scores))

    # ── Prepare pools for Step 4 batch ───────────────────────────────────────
    ibm_list = sorted(launder_senders)
    flagged_list = [s.account_id for s in all_scores if s.is_flagged]
    hn_ids = [r["account_id"] for r in gt_rows if r["gold_is_match"] == "False"]

    # Seeded shuffle of non-flagged accounts for conflict pool
    non_flagged_all = [s.account_id for s in all_scores if not s.is_flagged]
    _rng_nf = random.Random(EVAL_SEED ^ 0xDEAD)  # separate seed; doesn't consume main rng
    _rng_nf.shuffle(non_flagged_all)
    non_flagged_sample = non_flagged_all[:_NON_FLAGGED_SAMPLE]

    all_to_eval = sorted(
        set(ibm_list) | set(flagged_list) | set(hn_ids) | set(non_flagged_sample)
    )
    logger.info("[Step 6] Running Step 4 on %d accounts…", len(all_to_eval))
    all_firings = _batch_step4(df, all_to_eval, "batch")

    ibm_firings     = {a: all_firings.get(a, []) for a in ibm_list}
    flagged_firings = {a: all_firings.get(a, []) for a in flagged_list}
    hn_firings      = {a: all_firings.get(a, []) for a in hn_ids}
    sample_firings  = {a: all_firings.get(a, []) for a in non_flagged_sample}

    # ── Build slices ──────────────────────────────────────────────────────────
    used: set[str] = set()

    logger.info("[Step 6] Building sanctions_hit slice…")
    sh_cases  = _build_sanctions_hit(gt_rows, df_idx, used)

    logger.info("[Step 6] Building sanctions_near_miss slice…")
    snm_cases = _build_near_miss(gt_rows, df_idx, hn_firings, score_map, used, rng)

    logger.info("[Step 6] Building ibm_labeled slice…")
    ibm_cases = _build_ibm_labeled(df_idx, ibm_list, ibm_firings, used, rng)

    logger.info("[Step 6] Building typology slice…")
    typ_cases = _build_typology(df_idx, patterns, launder_senders, used, rng)

    logger.info("[Step 6] Building rules_anomaly_conflict slice…")
    conf_cases = _build_conflict(
        df_idx, flagged_firings, sample_firings, ibm_firings, score_map, used, rng
    )

    all_cases = sh_cases + snm_cases + ibm_cases + typ_cases + conf_cases

    # ── Final validation ──────────────────────────────────────────────────────
    if len(all_cases) != EVAL_SIZE:
        raise RuntimeError(f"Expected {EVAL_SIZE} cases, got {len(all_cases)}")

    acct_ids = [c.account_id for c in all_cases]
    dup_accts = [a for a, n in Counter(acct_ids).items() if n > 1]
    if dup_accts:
        raise RuntimeError(f"Duplicate account_ids in eval set: {dup_accts}")

    case_ids = [c.case_id for c in all_cases]
    dup_cases = [ci for ci, n in Counter(case_ids).items() if n > 1]
    if dup_cases:
        raise RuntimeError(f"Duplicate case_ids in eval set: {dup_cases}")

    non_overlay = [c.account_id for c in all_cases if c.account_id not in overlay_ids]
    if non_overlay:
        raise RuntimeError(f"Eval accounts not in identity overlay: {non_overlay}")

    return all_cases


def save_eval(
    cases: list[EvalCase],
    path: str | Path = _EVAL_OUT,
    checksums_path: str | Path = _CHECKSUM_FILE,
    force: bool = False,
) -> str:
    """
    Write cases to JSONL (one JSON object per line), then record SHA-256.
    Returns the hex digest.
    If *force* is True, removes any existing checksum entry before appending.
    """
    path = Path(path)
    checksums_path = Path(checksums_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write via temp file
    tmp = path.with_suffix(".tmp")
    content = "\n".join(c.model_dump_json() for c in cases) + "\n"
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)

    if force:
        _remove_checksum_line(checksums_path, str(path))

    append_checksum(path, checksums_path)
    return compute_sha256(path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_summary(cases: list[EvalCase], path: Path, digest: str = "") -> None:
    labels   = Counter(c.gold_label   for c in cases)
    types    = Counter(c.case_type    for c in cases)
    typos    = Counter(c.typology     for c in cases if c.typology)
    conf_sub = Counter(c.conflict_type for c in cases if c.conflict_type)
    bands    = Counter(c.severity_band for c in cases if c.severity_band is not None)
    print(f"\n{'='*64}")
    print(f"eval.jsonl: {path}  ({len(cases)} cases)")
    print(f"  Labels:          {dict(sorted(labels.items()))}")
    print(f"  Case types:      {dict(types)}")
    print(f"  Typologies:      {dict(sorted(typos.items()))}")
    print(f"  Conflict types:  {dict(conf_sub)}")
    print(f"  Severity bands:  {dict(sorted(bands.items()))}")
    if digest:
        print(f"  SHA-256:         {digest}")
    print(f"{'='*64}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and freeze data/fixtures/eval.jsonl"
    )
    parser.add_argument("--trans",        default=str(_TRANS_PATH))
    parser.add_argument("--overlay",      default=str(_OVERLAY_PATH))
    parser.add_argument("--ground-truth", default=str(_GT_PATH))
    parser.add_argument("--patterns",     default=str(_PATTERNS_PATH))
    parser.add_argument("--out",          default=str(_EVAL_OUT))
    parser.add_argument("--checksums",    default=str(_CHECKSUM_FILE))
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild and re-freeze even if eval.jsonl already exists",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"[SKIP] {out_path} already frozen. Use --force to rebuild.")
        cases: list[EvalCase] = []
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    cases.append(EvalCase.model_validate_json(line))
        _print_summary(cases, out_path)
        sys.exit(0)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Verify existing frozen artifacts are intact before building
    verify_checksums(args.checksums)
    logger.info("[Step 6] Existing checksums verified.")

    cases = build_eval_set(
        trans_path=args.trans,
        overlay_path=args.overlay,
        ground_truth_path=args.ground_truth,
        patterns_path=args.patterns,
    )

    digest = save_eval(
        cases,
        path=Path(args.out),
        checksums_path=Path(args.checksums),
        force=args.force,
    )
    _print_summary(cases, Path(args.out), digest)
    print("[Step 6] eval.jsonl frozen.")


if __name__ == "__main__":
    main()
