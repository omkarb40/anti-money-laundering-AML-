"""
Step 5 — Deterministic robust-z anomaly scorer.

Fully deterministic, explainable anomaly scoring method:

  1. For each feature, compute median and MAD (median absolute deviation).
  2. robust_z[i, j] = |x[i,j] - median[j]| / (1.4826 * MAD[j] + fallback_std[j])
     where fallback_std[j] = std(feature j) + _EPSILON handles zero-MAD columns
     (e.g. a binary feature that is 0.0 for all-but-one accounts).
  3. Per-feature robust-z is capped at _Z_CAP to prevent a single pathological
     feature from dominating when MAD ≈ 0.
  4. Composite score = mean of per-feature capped robust-z values.
  5. Higher composite score → more anomalous.
  6. Percentile = average-rank / N;  1.0 = most anomalous.
  7. is_flagged  = percentile >= ANOMALY_FLAGGING_PERCENTILE (frozen threshold).

No random state, no bootstrap, no model object — identical output on every call.

Score direction
---------------
score      : composite robust-z (non-negative float; higher = more anomalous)
percentile : [0.0, 1.0];  1.0 = most anomalous, 0.0 = most normal
is_flagged : percentile >= ANOMALY_FLAGGING_PERCENTILE (top 0.5 % by default)

Public API
----------
score_accounts(feature_df, log_path)  → list[AnomalyScore]
get_score(account_id, scores)         → Optional[AnomalyScore]
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from aml_copilot.schemas import AnomalyScore
from aml_copilot.step4_rules.thresholds import ANOMALY_FLAGGING_PERCENTILE
from aml_copilot.step5_anomaly.features import (
    EXCLUDED_FEATURES,
    FEATURE_COLS,
    log_excluded_features,
)

# ── Robust-z constants ────────────────────────────────────────────────────────

# Scale of the normal distribution captured by MAD: E[|X|] = σ * 0.6745
# → consistency factor 1 / 0.6745 ≈ 1.4826 makes MAD-based scale equal to σ.
_CONSISTENCY_FACTOR: float = 1.4826

# Guard against division by zero when MAD = 0 and std = 0.
_EPSILON: float = 1e-8

# Per-feature robust-z cap.  Prevents a constant-MAD feature (e.g. off_hours
# fraction that is 0.0 for 99% of accounts) from producing 1e10 z-scores for
# the rare non-zero accounts and swamping the composite mean.
_Z_CAP: float = 25.0


# ── Core computation ──────────────────────────────────────────────────────────

def _compute_robust_z_composite(X: np.ndarray) -> np.ndarray:
    """
    Compute the mean capped per-feature robust-z score for each row.

    Parameters
    ----------
    X : np.ndarray, shape (N, P)
        Feature matrix (float32 or float64; zero/NaN-free after build_feature_matrix).

    Returns
    -------
    np.ndarray, shape (N,), dtype float64
        Composite anomaly score per account.  Higher = more anomalous.
        Always finite (no NaN, no inf).
    """
    n = X.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)

    X64 = X.astype(np.float64)

    medians  = np.median(X64, axis=0)         # (P,)
    abs_dev  = np.abs(X64 - medians)           # (N, P)
    mads     = np.median(abs_dev, axis=0)      # (P,)
    stds     = X64.std(axis=0)                 # (P,)  population std

    # Prefer MAD-based scale; fall back to std when MAD = 0.
    scales = np.where(
        mads > 0,
        _CONSISTENCY_FACTOR * mads,
        stds + _EPSILON,
    )                                           # (P,)

    z_scores  = np.clip(abs_dev / scales, 0.0, _Z_CAP)   # (N, P)
    composite = z_scores.mean(axis=1)                      # (N,)
    return composite


def _compute_anomaly_percentile(scores: np.ndarray) -> np.ndarray:
    """
    Rank-based anomaly percentile: 1.0 = most anomalous, 0.0 = most normal.

    Higher composite score → higher rank → higher percentile.
    Ties receive the average rank of the tied group (average-rank method).

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
        Composite robust-z scores; higher = more anomalous.

    Returns
    -------
    np.ndarray, shape (N,), dtype float64
    """
    n = len(scores)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.array([1.0], dtype=np.float64)

    # np.unique returns values sorted ascending; inverse maps each element back.
    unique_vals, inverse = np.unique(scores, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique_vals)).astype(np.float64)

    # 1-indexed cumulative ranks
    ends      = np.cumsum(counts)          # last rank in each tie group
    starts    = ends - counts + 1.0       # first rank in each tie group
    avg_ranks = (starts + ends) / 2.0

    per_element = avg_ranks[inverse]
    return per_element / n


# ── Public API ────────────────────────────────────────────────────────────────

def score_accounts(
    feature_df: pl.DataFrame,
    log_path: str | Path | None = None,
) -> list[AnomalyScore]:
    """
    Score all accounts using deterministic robust-z anomaly scoring.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Output of build_feature_matrix().  Must contain FEATURE_COLS columns.
    log_path : optional
        If provided, excluded features are appended to this file via
        log_excluded_features() on every call.

    Returns
    -------
    list[AnomalyScore]
        One entry per account, sorted by account_id (alphabetical).
        Identical inputs always produce bitwise-identical output.
    """
    if log_path is not None:
        log_excluded_features(EXCLUDED_FEATURES, log_path)

    sorted_df    = feature_df.sort("account_id")
    account_ids: list[str] = sorted_df["account_id"].to_list()

    X              = sorted_df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    composite      = _compute_robust_z_composite(X)
    percentiles    = _compute_anomaly_percentile(composite)

    results: list[AnomalyScore] = []
    for i, account_id in enumerate(account_ids):
        pct = float(percentiles[i])
        results.append(
            AnomalyScore(
                account_id=account_id,
                score=float(composite[i]),
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
    """Linear-scan lookup helper for Step 7.

    Returns the AnomalyScore for account_id, or None if not found.
    """
    for s in scores:
        if s.account_id == account_id:
            return s
    return None
