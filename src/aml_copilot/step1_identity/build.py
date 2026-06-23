"""
Step 1 build command.

Usage:
    python -m aml_copilot.step1_identity.build

Reads:
    data/processed/accounts.parquet
    data/raw/ofac/sdn_advanced.xml
    data/raw/ofac/cons_advanced.xml  (optional)

Writes:
    data/processed/identity_overlay.parquet
    data/fixtures/ground_truth_matches.csv
    artifacts/checksums.sha256  (appended)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]  # project root: .../ML/


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Step 1 identity overlay")
    parser.add_argument(
        "--accounts", default=str(_ROOT / "data/processed/accounts.parquet"),
        help="Path to accounts.parquet (Step 0 output)",
    )
    parser.add_argument(
        "--sdn", default=str(_ROOT / "data/raw/ofac/sdn_advanced.xml"),
        help="Path to OFAC sdn_advanced.xml",
    )
    parser.add_argument(
        "--cons", default=str(_ROOT / "data/raw/ofac/cons_advanced.xml"),
        help="Path to OFAC cons_advanced.xml (optional)",
    )
    parser.add_argument(
        "--overlay-out", default=str(_ROOT / "data/processed/identity_overlay.parquet"),
    )
    parser.add_argument(
        "--gt-out", default=str(_ROOT / "data/fixtures/ground_truth_matches.csv"),
    )
    parser.add_argument(
        "--checksum-file", default=str(_ROOT / "artifacts/checksums.sha256"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    # Deferred imports so --help works without heavy deps
    from aml_copilot.step1_identity.ofac_reader import load_ofac_records
    from aml_copilot.step1_identity.overlay import build_identity_overlay, save_overlay
    from aml_copilot.step1_identity.ground_truth import save_ground_truth

    try:
        logger.info("Loading OFAC records from %s ...", args.sdn)
        cons = args.cons if Path(args.cons).exists() else None
        records = load_ofac_records(args.sdn, cons)
        individuals = [r for r in records if r.entry_type == "Individual"]
        logger.info(
            "OFAC loaded: %d total records (%d individuals)", len(records), len(individuals)
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error(
            "Download OFAC data from https://ofac.treasury.gov/sanctions-list-service "
            "and place sdn_advanced.xml under data/raw/ofac/"
        )
        return 1

    logger.info("Building identity overlay (seed=%d) ...", args.seed)
    overlay, fixture_rows = build_identity_overlay(
        accounts_parquet=args.accounts,
        ofac_records=records,
        seed=args.seed,
    )

    save_overlay(overlay, args.overlay_out)
    logger.info("Overlay: %d rows → %s", len(overlay), args.overlay_out)

    save_ground_truth(fixture_rows, args.gt_out, args.checksum_file)
    tp = sum(1 for r in fixture_rows if r["gold_is_match"])
    hn = sum(1 for r in fixture_rows if not r["gold_is_match"])
    logger.info("Ground truth: %d rows (%d TP / %d HN) → %s", len(fixture_rows), tp, hn, args.gt_out)

    # Verify no raw OFAC names leaked into overlay
    from aml_copilot.step1_identity.ofac_reader import build_raw_name_set
    raw_names = build_raw_name_set(records)
    overlay_names = set(overlay["name"].to_list())
    leaked = overlay_names & raw_names
    if leaked:
        logger.error("SAFETY VIOLATION: %d raw OFAC names found in overlay!", len(leaked))
        return 1
    logger.info("[OK] Safety check passed: no raw OFAC names in overlay name column")

    flavors = {}
    for r in fixture_rows:
        flavors[r["match_flavor"]] = flavors.get(r["match_flavor"], 0) + 1
    logger.info("Flavor distribution: %s", flavors)
    logger.info("Step 1 DoD: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
