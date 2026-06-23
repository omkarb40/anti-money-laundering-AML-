"""
Step 5 — IsolationForest anomaly scorer.

Score semantics
---------------
score      : raw decision_function output; more negative = more anomalous
percentile : anomaly percentile; 1.0 = most anomalous, 0.0 = most normal
is_flagged : percentile >= ANOMALY_FLAGGING_PERCENTILE (top 0.5% of population)

Public API
----------
fit_model(feature_df)                        → IsolationForest
score_accounts(feature_df, model, log_path)  → list[AnomalyScore]
get_score(account_id, scores)                → Optional[AnomalyScore]
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest

from aml_copilot.schemas import AnomalyScore
from aml_copilot.step4_rules.thresholds import (
    ANOMALY_CONTAMINATION,
    ANOMALY_FLAGGING_PERCENTILE,
)
from aml_copilot.step5_anomaly.features import (
    EXCLUDED_FEATURES,
    FEATURE_COLS,
    log_excluded_features,
)

# ── Module-level constants (not in frozen thresholds.py — implementation details)
ANOMALY_RANDOM_STATE: int = 42
ANOMALY_N_ESTIMATORS: int = 200


# ── Percentile computation ────────────────────────────────────────────────────

def _compute_anomaly_percentile(scores: np.ndarray) -> np.ndarray:
    """
    Anomaly percentile: 1.0 = most anomalous, 0.0 = most normal.

    IsolationForest decision_function: more negative = more anomalous.
    We negate before ranking so that higher rank maps to more anomalous.

    Tie handling: identical raw scores receive the same percentile via the
    average-rank method — (first_rank + last_rank) / 2 for each tie group,
    normalised by N.  This is O(N log N); one call on the full array.

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
        Raw decision_function output.

    Returns
    -------
    np.ndarray, shape (N,), dtype float64
        Anomaly percentile for each account, in the same order as scores.
    """
    n = len(scores)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.array([1.0], dtype=np.float64)

    neg = -scores  # higher neg = more anomalous = higher rank

    # np.unique returns unique values sorted ascending and an inverse index
    unique_neg, inverse = np.unique(neg, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique_neg)).astype(np.float64)

    # Cumulative 1-indexed ranks
    ends = np.cumsum(counts)         # last rank in each tie group
    starts = ends - counts + 1.0    # first rank in each tie group
    avg_ranks = (starts + ends) / 2.0

    per_element = avg_ranks[inverse]
    return per_element / n


# ── Public API ────────────────────────────────────────────────────────────────

def fit_model(feature_df: pl.DataFrame) -> IsolationForest:
    """
    Fit IsolationForest on feature_df.

    Rows are sorted alphabetically by account_id before fitting to guarantee
    that sklearn's internal bootstrap sampling is order-deterministic across
    calls with different DataFrame row orderings.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Output of build_feature_matrix().  Must contain FEATURE_COLS columns.

    Returns
    -------
    sklearn.ensemble.IsolationForest
        Fitted model; pass to score_accounts() to obtain AnomalyScore objects.
    """
    sorted_df = feature_df.sort("account_id")
    X = sorted_df.select(FEATURE_COLS).to_numpy().astype(np.float32)

    model = IsolationForest(
        n_estimators=ANOMALY_N_ESTIMATORS,
        contamination=ANOMALY_CONTAMINATION,
        random_state=ANOMALY_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def score_accounts(
    feature_df: pl.DataFrame,
    model: IsolationForest,
    log_path: str | Path | None = None,
) -> list[AnomalyScore]:
    """
    Score all accounts and return AnomalyScore objects.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Output of build_feature_matrix().
    model : IsolationForest
        Output of fit_model().
    log_path : optional
        If provided, excluded features are appended to this file via
        log_excluded_features().

    Returns
    -------
    list[AnomalyScore]
        One entry per account, sorted by account_id (alphabetical).
    """
    if log_path is not None:
        log_excluded_features(EXCLUDED_FEATURES, log_path)

    sorted_df = feature_df.sort("account_id")
    account_ids: list[str] = sorted_df["account_id"].to_list()

    X = sorted_df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    raw_scores: np.ndarray = model.decision_function(X)
    percentiles: np.ndarray = _compute_anomaly_percentile(raw_scores)

    results: list[AnomalyScore] = []
    for i, account_id in enumerate(account_ids):
        pct = float(percentiles[i])
        results.append(
            AnomalyScore(
                account_id=account_id,
                score=float(raw_scores[i]),
                percentile=pct,
                is_flagged=bool(pct >= ANOMALY_FLAGGING_PERCENTILE),
                excluded_features=list(EXCLUDED_FEATURES),
            )
        )
    return results


def get_score(
    account_id: str,
    scores: list[AnomalyScore],
) -> Optional[AnomalyScore]:
    """
    Linear-scan lookup helper for Step 7.

    Returns the AnomalyScore for account_id, or None if not found.
    """
    for s in scores:
        if s.account_id == account_id:
            return s
    return None
