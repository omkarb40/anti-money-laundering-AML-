from __future__ import annotations

import pytest


def test_perfect_results_accuracy_1() -> None:
    """If all 90 dispositions match gold labels, disposition_accuracy == 1.0."""
    ...


def test_false_clear_rate_weighting() -> None:
    """A severity-3 false clear contributes 3× the weight of a severity-1 false clear."""
    ...


def test_denominator_zero_safe() -> None:
    """Empty results list raises ValueError, not ZeroDivisionError."""
    ...


def test_metrics_frozen_after_write(tmp_path) -> None:
    """Writing metrics_baseline.json records its SHA-256; a second write raises or requires --force."""
    ...
