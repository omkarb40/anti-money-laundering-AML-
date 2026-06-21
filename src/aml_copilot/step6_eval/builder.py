from __future__ import annotations

import polars as pl

from aml_copilot.schemas import EvalCase

EVAL_SEED: int = 42
EVAL_SIZE: int = 90

SLICE_COUNTS: dict[str, int] = {
    "ibm_labeled": 30,
    "sanctions_hit": 15,
    "sanctions_near_miss": 15,
    "rules_anomaly_conflict": 10,
    "typology": 20,
}


def build_eval_set(
    transactions: pl.DataFrame,
    ground_truth_path: str,
    patterns: dict[str, list[str]],
) -> list[EvalCase]:
    """
    Assemble 90 EvalCase objects from the 5 slices defined in SLICE_COUNTS.
    Selection within each slice is seeded with EVAL_SEED for reproducibility.
    Gold labels are set from IBM ground truth — never from system output.
    """
    ...


def save_eval(cases: list[EvalCase], path: str) -> None:
    """Write cases to JSONL at path (one EvalCase JSON per line), then record SHA-256."""
    ...
