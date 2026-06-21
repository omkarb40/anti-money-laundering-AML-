from __future__ import annotations

import polars as pl

from aml_copilot.schemas import AnomalyScore


def fit_model(feature_matrix: pl.DataFrame):
    """Fit IsolationForest on feature_matrix. Returns the fitted sklearn model."""
    ...


def score_accounts(feature_matrix: pl.DataFrame, model) -> list[AnomalyScore]:
    """
    Run model.decision_function on feature_matrix.
    Compute per-account percentile rank; set is_flagged based on ANOMALY_FLAGGING_PERCENTILE.
    Each AnomalyScore.excluded_features reflects what was dropped before this call.
    """
    ...
