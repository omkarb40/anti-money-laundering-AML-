from __future__ import annotations

import polars as pl

# These column names must never appear in the feature matrix.
# Any feature derived from the laundering label or net-flow accounting leaks the target.
EXCLUDED_FEATURES: list[str] = [
    "running_balance_delta",
    "cumulative_net_flow",
    "total_in_minus_out",
    "is_laundering",
]

PERMITTED_FEATURES: list[str] = [
    "txn_count_7d",
    "txn_count_30d",
    "amount_mean",
    "amount_std",
    "amount_max",
    "amount_skew",
    "counterparty_diversity",
    "round_number_ratio",
    "time_of_day_entropy",
    "off_hours_fraction",
]


def build_feature_matrix(
    df: pl.DataFrame,
    accounts: pl.DataFrame,
) -> pl.DataFrame:
    """
    Return a DataFrame with one row per account and columns == PERMITTED_FEATURES.
    EXCLUDED_FEATURES must not appear at any intermediate stage.
    """
    ...


def log_excluded_features(excluded: list[str], log_path: str) -> None:
    """Append one log line per excluded feature with timestamp to log_path."""
    ...
