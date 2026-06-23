"""
Write and freeze ground_truth_matches.csv.

This module handles the write-once lifecycle: first call writes the CSV and
records its SHA-256 in artifacts/checksums.sha256. Any subsequent call raises
RuntimeError to prevent silent overwrite after the fixture is frozen.

SECURITY: ofac_canonical_name values are written to CSV only. They must not
appear in log output at INFO level or above.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from aml_copilot.schemas import GroundTruthRow
from aml_copilot.utils.checksum import append_checksum

logger = logging.getLogger(__name__)

_FIELDNAMES = [
    "row_id",
    "account_id",
    "assigned_name",
    "ofac_uid",
    "ofac_canonical_name",
    "match_flavor",
    "expected_score_min",
    "expected_score_max",
    "gold_is_match",
]


def save_ground_truth(
    rows: list[dict],
    output_path: str | Path,
    checksum_file: str | Path,
) -> None:
    """
    Validate, write, and freeze ground_truth_matches.csv.

    Raises
    ------
    RuntimeError
        If a checksum entry for output_path already exists (write-once guard).
    ValueError
        If rows fail schema validation.
    """
    output_path = Path(output_path)
    checksum_file = Path(checksum_file)

    # Validate all rows against Pydantic schema before touching the filesystem
    validated: list[GroundTruthRow] = []
    for row in rows:
        validated.append(GroundTruthRow(**row))

    if len(validated) != 50:
        raise ValueError(f"Expected exactly 50 ground-truth rows; got {len(validated)}")

    tp_count = sum(1 for r in validated if r.gold_is_match)
    hn_count = sum(1 for r in validated if not r.gold_is_match)
    if tp_count != 20 or hn_count != 30:
        raise ValueError(f"Expected 20 TPs + 30 HNs; got {tp_count} TPs + {hn_count} HNs")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for r in validated:
            writer.writerow({
                "row_id": r.row_id,
                "account_id": r.account_id,
                "assigned_name": r.assigned_name,
                "ofac_uid": r.ofac_uid,
                "ofac_canonical_name": r.ofac_canonical_name,  # sensitive — CSV only
                "match_flavor": r.match_flavor,
                "expected_score_min": r.expected_score_min,
                "expected_score_max": r.expected_score_max,
                "gold_is_match": r.gold_is_match,
            })

    # Freeze: raises RuntimeError if already checksummed
    digest = append_checksum(output_path, checksum_file)
    logger.info(
        "ground_truth_matches.csv frozen: %d rows (%d TP / %d HN) [SHA-256: %s...]",
        len(validated), tp_count, hn_count, digest[:12],
    )


def load_ground_truth(path: str | Path) -> list[GroundTruthRow]:
    """Read and validate an existing (frozen) ground_truth_matches.csv."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ground_truth_matches.csv not found: {path}")

    rows: list[GroundTruthRow] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for record in csv.DictReader(fh):
            rows.append(GroundTruthRow(
                row_id=int(record["row_id"]),
                account_id=record["account_id"],
                assigned_name=record["assigned_name"],
                ofac_uid=record["ofac_uid"],
                ofac_canonical_name=record["ofac_canonical_name"],
                match_flavor=record["match_flavor"],  # type: ignore[arg-type]
                expected_score_min=float(record["expected_score_min"]),
                expected_score_max=float(record["expected_score_max"]),
                gold_is_match=record["gold_is_match"].lower() == "true",
            ))
    return rows
