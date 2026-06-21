from __future__ import annotations

import polars as pl

from aml_copilot.schemas import GroundTruthRow

TP_COUNT: int = 20        # 5 per flavor × 4 flavors
HN_COUNT: int = 30        # hard negatives
HN_JW_MIN: float = 0.80  # hard negative Jaro-Winkler lower bound
HN_JW_MAX: float = 0.88  # hard negative Jaro-Winkler upper bound — must stay < 0.90


def build_ground_truth(
    overlay: pl.DataFrame,
    ofac_index: dict,
) -> list[GroundTruthRow]:
    """
    Build 20 true positives (5 each: exact, transliteration, typo_ocr, partial_reorder)
    and 30 hard negatives in a single pass.
    Hard negatives share a surname with a real OFAC entry; JW score must be 0.80–0.88.
    """
    ...


def save_ground_truth(rows: list[GroundTruthRow], path: str) -> None:
    """Write ground truth to CSV at path, then record its SHA-256 in checksums file."""
    ...
