"""
Step 5 — Anomaly feature engineering.

Computes 15 account-level, non-leaky features from HI-Small transaction data
using Polars group_by aggregations (no Python row iteration, no pandas).

EXCLUDED_FEATURES are logged on every run and must never appear in the output.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# ── Exclusion list ─────────────────────────────────────────────────────────────
# Derived features that encode the IBM simulator label or net-flow accounting
# are explicitly enumerated here.  Extending this list after Step 6 eval set
# construction would constitute retroactive feature selection — treat as frozen.

EXCLUDED_FEATURES: list[str] = [
    "is_laundering",                    # direct IBM label
    "pattern_label",                    # HI-Small_Patterns.txt — functionally a label
    "net_flow",                         # sum(in) - sum(out) encodes pass-through typology
    "balance_delta",                    # alias for net_flow
    "total_in_minus_out",               # alias for net_flow
    "running_balance_delta",            # cumulative variant of net_flow
    "cumulative_net_flow",              # cumulative variant of net_flow
    "name",                             # Faker-assigned; zero informational signal
    "kyc_risk",                         # Faker-assigned; zero informational signal
    "receiving_currency_concentration", # IBM may have used currencies systematically
    "payment_currency_concentration",   # same concern; excluded out of caution
]

# ── Feature columns (output schema, ordered) ───────────────────────────────────
FEATURE_COLS: list[str] = [
    # Transaction count
    "outbound_count",       # total outbound transactions
    "inbound_count",        # total inbound transactions
    "in_out_ratio",         # inbound_count / (outbound_count + 1)
    # Amount statistics (outbound only)
    "amount_mean",          # mean outbound amount_paid
    "amount_std",           # std of outbound amount_paid
    "amount_max",           # max outbound amount_paid
    "amount_skew",          # Fisher skewness of outbound amount_paid
    "round_amount_ratio",   # fraction of outbound amounts divisible by 1000
    # Counterparty diversity
    "unique_out_recipients",# count distinct to_account (outbound)
    "out_diversity",        # unique_out_recipients / (outbound_count + 1)
    "unique_in_senders",    # count distinct from_account (inbound)
    "in_diversity",         # unique_in_senders / (inbound_count + 1)
    # Temporal
    "off_hours_fraction",   # fraction of appearances outside 07:00–20:00
    "txn_hour_entropy",     # Shannon entropy (base-2) of hour-of-day distribution
    "avg_daily_velocity",   # (outbound_count + inbound_count) / (day_range + 1)
]


# ── Public API ─────────────────────────────────────────────────────────────────

def build_feature_matrix(df: pl.DataFrame, accounts: pl.DataFrame) -> pl.DataFrame:
    """
    Build a 15-feature matrix with one row per account.

    Parameters
    ----------
    df : pl.DataFrame
        Full transaction DataFrame from Step 0.  May contain is_laundering;
        that column is never selected, read, or used at any stage.
    accounts : pl.DataFrame
        Single-column DataFrame with column "account_id".  Left-join anchor —
        every account_id in this DataFrame appears in the output, even if it
        has no transactions (those accounts receive all-zero features).

    Returns
    -------
    pl.DataFrame
        Columns: account_id + FEATURE_COLS (all float32), sorted by account_id.
    """
    # Select only the columns needed — never touch EXCLUDED_FEATURES
    _need = ["timestamp", "from_account", "to_account", "amount_paid"]
    df_clean = df.select([c for c in _need if c in df.columns])

    # Sanity check: forbidden columns must not have slipped into the clean copy
    for col in EXCLUDED_FEATURES:
        if col in df_clean.columns:
            raise ValueError(
                f"Forbidden column {col!r} reached the feature builder — leakage bug."
            )

    # ── Add hour-of-day for temporal aggregations ─────────────────────────────
    df_h = df_clean.with_columns(pl.col("timestamp").dt.hour().alias("_hour"))

    # ── Outbound: from_account perspective ───────────────────────────────────
    outbound_feats = (
        df_h.group_by("from_account")
        .agg([
            pl.len().alias("outbound_count"),
            pl.col("to_account").n_unique().alias("unique_out_recipients"),
            pl.col("amount_paid").mean().alias("amount_mean"),
            pl.col("amount_paid").std(ddof=1).alias("amount_std"),
            pl.col("amount_paid").max().alias("amount_max"),
            pl.col("amount_paid").skew().alias("amount_skew"),
            # Null for groups of n<3; filled later with 0.
            ((pl.col("amount_paid").round(0) % 1000.0) == 0.0)
            .mean()
            .alias("round_amount_ratio"),
        ])
        .rename({"from_account": "account_id"})
    )

    # ── Inbound: to_account perspective ──────────────────────────────────────
    inbound_feats = (
        df_h.group_by("to_account")
        .agg([
            pl.len().alias("inbound_count"),
            pl.col("from_account").n_unique().alias("unique_in_senders"),
        ])
        .rename({"to_account": "account_id"})
    )

    # ── Stacked view: account appears as sender OR receiver ───────────────────
    # Fraction-based temporal features are unaffected by the 2× double-count
    # (numerator and denominator scale equally).  avg_daily_velocity reflects
    # total transaction involvement per day regardless of direction.
    stacked = pl.concat([
        df_h.select(
            pl.col("from_account").alias("account_id"),
            pl.col("_hour"),
            pl.col("timestamp"),
        ),
        df_h.select(
            pl.col("to_account").alias("account_id"),
            pl.col("_hour"),
            pl.col("timestamp"),
        ),
    ])

    # Day range: span from first to last appearance in either role
    velocity_feats = (
        stacked.group_by("account_id")
        .agg([
            pl.col("timestamp").max().alias("_ts_max"),
            pl.col("timestamp").min().alias("_ts_min"),
        ])
        .with_columns([
            (
                (pl.col("_ts_max") - pl.col("_ts_min"))
                .dt.total_seconds()
                .cast(pl.Float64)
                / 86400.0
            ).alias("_day_range"),
        ])
    )

    # Off-hours fraction: transactions outside 07:00–20:00
    off_hours_feats = (
        stacked.group_by("account_id")
        .agg([
            ((pl.col("_hour") < 7) | (pl.col("_hour") > 20))
            .mean()
            .alias("off_hours_fraction"),
        ])
    )

    # Hour-of-day entropy (Shannon, base-2), two-step computation:
    # 1) count appearances per (account, hour)
    hour_counts = (
        stacked.group_by(["account_id", "_hour"])
        .agg(pl.len().alias("_h_cnt"))
    )
    # 2) join total appearance count, compute probability, then entropy contribution
    hour_totals = stacked.group_by("account_id").agg(pl.len().alias("_h_total"))
    entropy_feats = (
        hour_counts.join(hour_totals, on="account_id")
        .with_columns(
            (
                pl.col("_h_cnt").cast(pl.Float64)
                / pl.col("_h_total").cast(pl.Float64)
            ).alias("_prob")
        )
        .with_columns(
            # -p * log2(p); prob is always > 0 here (comes from positive counts)
            (-pl.col("_prob") * pl.col("_prob").log(base=2.0)).alias("_h_entr")
        )
        .group_by("account_id")
        .agg(pl.col("_h_entr").sum().alias("txn_hour_entropy"))
    )

    # ── Join all feature blocks onto the accounts anchor ──────────────────────
    feat_df = (
        accounts
        .join(outbound_feats, on="account_id", how="left")
        .join(inbound_feats, on="account_id", how="left")
        .join(
            velocity_feats.select(["account_id", "_day_range"]),
            on="account_id",
            how="left",
        )
        .join(off_hours_feats, on="account_id", how="left")
        .join(entropy_feats, on="account_id", how="left")
    )

    # ── Fill count/flag features with 0 for accounts with no transactions ─────
    _zero_fill = [
        "outbound_count", "inbound_count",
        "unique_out_recipients", "unique_in_senders",
        "round_amount_ratio", "off_hours_fraction", "txn_hour_entropy",
    ]
    feat_df = feat_df.with_columns([
        pl.col(c).fill_null(0).cast(pl.Float64)
        for c in _zero_fill
        if c in feat_df.columns
    ])
    feat_df = feat_df.with_columns(pl.col("_day_range").fill_null(0.0))

    # ── Derived ratio features (computed after null-fill so denominators are 0)
    feat_df = feat_df.with_columns([
        (pl.col("inbound_count") / (pl.col("outbound_count") + 1.0))
        .alias("in_out_ratio"),
        (pl.col("unique_out_recipients") / (pl.col("outbound_count") + 1.0))
        .alias("out_diversity"),
        (pl.col("unique_in_senders") / (pl.col("inbound_count") + 1.0))
        .alias("in_diversity"),
        (
            (pl.col("outbound_count") + pl.col("inbound_count"))
            / (pl.col("_day_range") + 1.0)
        ).alias("avg_daily_velocity"),
    ])

    # ── Fill null amount stats with column median ─────────────────────────────
    # Accounts with no outbound transactions have null amount_* from the left join.
    # Accounts with a single outbound transaction have null amount_std / amount_skew
    # (undefined for n < 2 or n < 3 respectively).
    for col in ("amount_mean", "amount_std", "amount_max", "amount_skew"):
        if col in feat_df.columns:
            median_val = feat_df[col].median()
            fill_val = float(median_val) if median_val is not None else 0.0
            feat_df = feat_df.with_columns(pl.col(col).fill_null(fill_val))

    # ── Fill any residual NaN (e.g. 0/0 in derived features) with 0 ──────────
    present = [c for c in FEATURE_COLS if c in feat_df.columns]
    feat_df = feat_df.with_columns([pl.col(c).fill_nan(0.0) for c in present])

    # ── Select, order, cast, sort ─────────────────────────────────────────────
    feat_df = (
        feat_df
        .select(["account_id"] + FEATURE_COLS)
        .with_columns([pl.col(c).cast(pl.Float32) for c in FEATURE_COLS])
        .sort("account_id")
    )

    # Final guard: no forbidden columns in output
    for col in EXCLUDED_FEATURES:
        if col in feat_df.columns:
            raise RuntimeError(
                f"Leaky feature {col!r} survived into feature matrix — leakage bug."
            )

    return feat_df


def log_excluded_features(excluded: list[str], log_path: str | Path) -> None:
    """
    Append one timestamped log line per excluded feature to log_path.
    Creates parent directories if needed.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as fh:
        for feat in excluded:
            fh.write(f"{ts}  EXCLUDED_FEATURE: {feat}\n")
    logger.info("Logged %d excluded features to %s", len(excluded), path)
