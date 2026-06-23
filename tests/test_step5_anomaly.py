"""
Unit and integration tests for Step 5: anomaly feature engineering and scoring.

Unit tests use synthetic Polars DataFrames — no disk I/O required.
Integration tests require data/raw/HI-Small_Trans.csv and are auto-skipped
when that file is absent.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from aml_copilot.schemas import AnomalyScore
from aml_copilot.step4_rules.thresholds import ANOMALY_FLAGGING_PERCENTILE
from aml_copilot.step5_anomaly.features import (
    EXCLUDED_FEATURES,
    FEATURE_COLS,
    build_feature_matrix,
    log_excluded_features,
)
from aml_copilot.step5_anomaly.scorer import (
    ANOMALY_N_ESTIMATORS,
    ANOMALY_RANDOM_STATE,
    _compute_anomaly_percentile,
    fit_model,
    get_score,
    score_accounts,
)

# ── Helpers / local fixtures ───────────────────────────────────────────────────

_BASE = datetime(2022, 1, 1, 10, 0, 0)


def _ts(hour_offset: float) -> datetime:
    return _BASE + timedelta(hours=hour_offset)


def _make_tiny_df(n_legit: int = 10, n_launder: int = 5) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Tiny synthetic transaction DataFrame and accounts DataFrame for fast unit tests.
    Includes is_laundering so we can verify it's never used.
    """
    rows: list[dict] = []
    legit = [f"LEG{i:02d}" for i in range(n_legit)]
    launder = [f"LAU{i:02d}" for i in range(n_launder)]

    for i, acct in enumerate(legit):
        rows.append({
            "timestamp": _ts(float(i * 2)),
            "from_account": acct,
            "to_account": f"COUNTER{i:02d}",
            "amount_paid": float(100 + i * 50),
            "amount_received": 0.0,
            "is_laundering": 0,
        })

    for i, acct in enumerate(launder):
        for j in range(3):
            rows.append({
                "timestamp": _ts(float(i + j * 0.1)),
                "from_account": acct,
                "to_account": f"SHELL{i}{j:02d}",
                "amount_paid": 9000.0,
                "amount_received": 0.0,
                "is_laundering": 1,
            })

    df = pl.DataFrame(rows).sort("timestamp")
    all_ids = sorted(set(df["from_account"].to_list() + df["to_account"].to_list()))
    accounts_df = pl.DataFrame({"account_id": all_ids})
    return df, accounts_df


def _make_clear_outlier_feature_df(n_normal: int = 30) -> pl.DataFrame:
    """
    Feature DataFrame with one obvious outlier (all features = 1000) and
    n_normal normal accounts (all features = 1.0).  Built directly, bypassing
    build_feature_matrix so scorer tests are independent of feature engineering.
    """
    n = n_normal + 1
    outlier_vals = {col: [1000.0] + [1.0] * n_normal for col in FEATURE_COLS}
    return pl.DataFrame({
        "account_id": ["OUTLIER"] + [f"NORM{i:02d}" for i in range(n_normal)],
        **outlier_vals,
    }).with_columns([pl.col(c).cast(pl.Float32) for c in FEATURE_COLS])


def _make_synth_transactions(
    n_legit: int = 100,
    n_launder: int = 100,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """
    200-account synthetic dataset with clear anomaly separation.

    Legitimate accounts : 2–5 varied transactions in business hours
    Laundering-like     : 15–20 fan-out transactions at 2 am (structuring amounts)

    Returns (transactions_df, accounts_df, laundering_account_ids)
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    legit_ids = [f"LEGIT_{i:04d}" for i in range(n_legit)]
    launder_ids = [f"LAUND_{i:04d}" for i in range(n_launder)]

    for acct in legit_ids:
        n_txns = int(rng.integers(2, 6))
        for _ in range(n_txns):
            ts = _BASE + timedelta(
                days=int(rng.integers(0, 30)),
                hours=int(rng.integers(8, 18)),
                minutes=int(rng.integers(0, 60)),
            )
            rows.append({
                "timestamp": ts,
                "from_account": acct,
                "to_account": f"COUNTER_{rng.integers(0, 50):04d}",
                "amount_paid": float(rng.uniform(100, 5000)),
                "amount_received": 0.0,
                "is_laundering": 0,
            })

    for i, acct in enumerate(launder_ids):
        n_txns = int(rng.integers(15, 21))
        for j in range(n_txns):
            # Burst in 12 hours, 2 am off-hours
            ts = _BASE.replace(hour=2) + timedelta(minutes=float(j * 35))
            rows.append({
                "timestamp": ts,
                "from_account": acct,
                "to_account": f"SHELL_{rng.integers(0, 1000):04d}",
                "amount_paid": float(
                    rng.choice([9000.0, 9200.0, 8800.0, 9500.0, 8000.0])
                ),
                "amount_received": 0.0,
                "is_laundering": 1,
            })

    df = pl.DataFrame(rows).sort("timestamp")
    # Only labeled accounts in the model: counterparties (SHELL_*, COUNTER_*) are
    # unlabeled and would dominate the unsupervised population if included, causing
    # IsolationForest to treat both senders equally anomalous vs. the shell majority.
    labeled_accounts_df = pl.DataFrame({"account_id": legit_ids + launder_ids})
    return df, labeled_accounts_df, launder_ids


# ── TestBuildFeatureMatrix ─────────────────────────────────────────────────────

class TestBuildFeatureMatrix:
    def test_excluded_features_absent(self, tiny_transactions, tiny_accounts) -> None:
        """Feature matrix column names contain none of the EXCLUDED_FEATURES names."""
        feat_df = build_feature_matrix(tiny_transactions, tiny_accounts)
        for col in EXCLUDED_FEATURES:
            assert col not in feat_df.columns, (
                f"Excluded feature {col!r} appeared in feature matrix"
            )

    def test_no_label_in_features(self, tiny_transactions, tiny_accounts) -> None:
        """is_laundering is not present in the feature DataFrame at any pipeline stage."""
        assert "is_laundering" in tiny_transactions.columns, (
            "Precondition: tiny_transactions must contain is_laundering"
        )
        feat_df = build_feature_matrix(tiny_transactions, tiny_accounts)
        assert "is_laundering" not in feat_df.columns

    def test_no_net_flow_or_balance_delta(self, tiny_transactions, tiny_accounts) -> None:
        feat_df = build_feature_matrix(tiny_transactions, tiny_accounts)
        forbidden = {"net_flow", "balance_delta", "total_in_minus_out",
                     "running_balance_delta", "cumulative_net_flow"}
        for col in forbidden:
            assert col not in feat_df.columns

    def test_feature_count_is_15(self, tiny_transactions, tiny_accounts) -> None:
        feat_df = build_feature_matrix(tiny_transactions, tiny_accounts)
        feature_cols = [c for c in feat_df.columns if c != "account_id"]
        assert len(feature_cols) == 15
        assert set(feature_cols) == set(FEATURE_COLS)

    def test_output_row_count_matches_accounts(self, tiny_transactions, tiny_accounts) -> None:
        feat_df = build_feature_matrix(tiny_transactions, tiny_accounts)
        assert len(feat_df) == len(tiny_accounts)

    def test_feature_dtypes_float32(self, tiny_transactions, tiny_accounts) -> None:
        feat_df = build_feature_matrix(tiny_transactions, tiny_accounts)
        for col in FEATURE_COLS:
            dtype = feat_df[col].dtype
            assert dtype == pl.Float32, f"Column {col!r} has dtype {dtype}, expected Float32"

    def test_sorted_by_account_id(self, tiny_transactions, tiny_accounts) -> None:
        feat_df = build_feature_matrix(tiny_transactions, tiny_accounts)
        ids = feat_df["account_id"].to_list()
        assert ids == sorted(ids)

    def test_accounts_with_no_transactions_get_zero_features(
        self, tiny_transactions
    ) -> None:
        """Phantom accounts (not in any transaction) receive all-zero features."""
        phantom = pl.DataFrame({"account_id": ["PHANTOM_XYZ"]})
        feat_df = build_feature_matrix(tiny_transactions, phantom)
        assert len(feat_df) == 1
        row = feat_df.to_dicts()[0]
        for col in ("outbound_count", "inbound_count", "txn_hour_entropy"):
            assert row[col] == pytest.approx(0.0), f"{col} expected 0 for phantom account"

    def test_no_nan_in_output(self, tiny_transactions, tiny_accounts) -> None:
        feat_df = build_feature_matrix(tiny_transactions, tiny_accounts)
        for col in FEATURE_COLS:
            null_count = feat_df[col].is_nan().sum() + feat_df[col].is_null().sum()
            assert null_count == 0, f"Column {col!r} has {null_count} null/NaN values"

    def test_outbound_count_correct(self) -> None:
        df = pl.DataFrame({
            "timestamp": [_ts(0), _ts(1), _ts(2)],
            "from_account": ["ACC", "ACC", "OTHER"],
            "to_account": ["B", "C", "ACC"],
            "amount_paid": [100.0, 200.0, 300.0],
            "amount_received": [0.0, 0.0, 300.0],
            "is_laundering": pl.Series([0, 0, 0], dtype=pl.Int8),
        })
        accounts = pl.DataFrame({"account_id": ["ACC"]})
        feat = build_feature_matrix(df, accounts)
        row = feat.to_dicts()[0]
        assert row["outbound_count"] == pytest.approx(2.0)
        assert row["inbound_count"] == pytest.approx(1.0)

    def test_round_amount_ratio(self) -> None:
        df = pl.DataFrame({
            "timestamp": [_ts(0), _ts(1), _ts(2), _ts(3)],
            "from_account": ["ACC", "ACC", "ACC", "ACC"],
            "to_account": ["B", "C", "D", "E"],
            "amount_paid": [1000.0, 2000.0, 500.0, 750.0],  # 2 round, 2 not
            "amount_received": [0.0, 0.0, 0.0, 0.0],
            "is_laundering": pl.Series([0, 0, 0, 0], dtype=pl.Int8),
        })
        accounts = pl.DataFrame({"account_id": ["ACC"]})
        feat = build_feature_matrix(df, accounts)
        row = feat.to_dicts()[0]
        assert row["round_amount_ratio"] == pytest.approx(0.5, rel=1e-3)

    def test_off_hours_fraction(self) -> None:
        # Two transactions at 2 am (off-hours), two at 10 am (in-hours)
        df = pl.DataFrame({
            "timestamp": [
                datetime(2022, 1, 1, 2, 0),   # off-hours
                datetime(2022, 1, 1, 3, 0),   # off-hours
                datetime(2022, 1, 1, 10, 0),  # in-hours
                datetime(2022, 1, 1, 14, 0),  # in-hours
            ],
            "from_account": ["ACC", "ACC", "ACC", "ACC"],
            "to_account": ["B", "C", "D", "E"],
            "amount_paid": [100.0, 100.0, 100.0, 100.0],
            "amount_received": [0.0, 0.0, 0.0, 0.0],
            "is_laundering": pl.Series([0, 0, 0, 0], dtype=pl.Int8),
        })
        accounts = pl.DataFrame({"account_id": ["ACC"]})
        feat = build_feature_matrix(df, accounts)
        row = feat.to_dicts()[0]
        # 2 out of 4 outbound txns (stacked appears 4 times total since no inbound)
        assert row["off_hours_fraction"] == pytest.approx(0.5, rel=1e-2)


# ── TestComputeAnomalyPercentile ───────────────────────────────────────────────

class TestComputeAnomalyPercentile:
    def test_direction(self) -> None:
        """More negative score (more anomalous) → higher percentile."""
        scores = np.array([1.0, 0.0, -1.0, -2.0])
        pcts = _compute_anomaly_percentile(scores)
        # -2.0 most anomalous → highest percentile; 1.0 most normal → lowest
        assert pcts[3] > pcts[2] > pcts[1] > pcts[0]

    def test_tie_same_percentile(self) -> None:
        """Identical scores → identical anomaly percentile (average-rank method)."""
        # scores: two tied normal accounts, one anomalous
        scores = np.array([-2.0, 0.0, 0.0])
        pcts = _compute_anomaly_percentile(scores)
        assert pcts[1] == pytest.approx(pcts[2]), (
            f"Tied accounts have different percentiles: {pcts[1]:.6f} != {pcts[2]:.6f}"
        )
        assert pcts[0] > pcts[1]  # anomalous > tied normal

    def test_tie_average_rank_values(self) -> None:
        """Verify exact percentile values from average-rank formula."""
        scores = np.array([-2.0, 0.0, 0.0])
        pcts = _compute_anomaly_percentile(scores)
        # neg = [2.0, 0.0, 0.0]
        # unique_neg = [0.0, 2.0]; counts = [2, 1]
        # ends = [2, 3]; starts = [1, 3]; avg_ranks = [1.5, 3.0]
        # element ranks: [3.0, 1.5, 1.5] → percentile [1.0, 0.5, 0.5]
        assert pcts[0] == pytest.approx(1.0)
        assert pcts[1] == pytest.approx(0.5)
        assert pcts[2] == pytest.approx(0.5)

    def test_monotone_no_ties(self) -> None:
        scores = np.array([3.0, 1.0, -1.0, -3.0])
        pcts = _compute_anomaly_percentile(scores)
        assert pcts[0] < pcts[1] < pcts[2] < pcts[3]

    def test_all_in_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.standard_normal(100)
        pcts = _compute_anomaly_percentile(scores)
        assert np.all(pcts >= 0.0) and np.all(pcts <= 1.0)

    def test_single_element(self) -> None:
        pcts = _compute_anomaly_percentile(np.array([-5.0]))
        assert pcts[0] == pytest.approx(1.0)

    def test_empty_array(self) -> None:
        pcts = _compute_anomaly_percentile(np.array([]))
        assert len(pcts) == 0

    def test_all_tied(self) -> None:
        scores = np.array([0.5, 0.5, 0.5])
        pcts = _compute_anomaly_percentile(scores)
        assert pcts[0] == pytest.approx(pcts[1]) == pytest.approx(pcts[2])


# ── TestFitModel ───────────────────────────────────────────────────────────────

class TestFitModel:
    def test_returns_isolation_forest(self) -> None:
        from sklearn.ensemble import IsolationForest as _IF
        feat_df = _make_clear_outlier_feature_df(n_normal=20)
        model = fit_model(feat_df)
        assert isinstance(model, _IF)

    def test_model_is_fitted(self) -> None:
        from sklearn.utils.validation import check_is_fitted
        feat_df = _make_clear_outlier_feature_df(n_normal=20)
        model = fit_model(feat_df)
        check_is_fitted(model)  # raises NotFittedError if not fitted

    def test_deterministic_across_row_orderings(self) -> None:
        """Shuffled input rows → identical model (decision_function output matches)."""
        feat_df = _make_clear_outlier_feature_df(n_normal=20)
        shuffled = feat_df.sample(fraction=1.0, shuffle=True, seed=99)

        m1 = fit_model(feat_df)
        m2 = fit_model(shuffled)

        X = feat_df.sort("account_id").select(FEATURE_COLS).to_numpy().astype(np.float32)
        scores1 = m1.decision_function(X)
        scores2 = m2.decision_function(X)

        np.testing.assert_array_almost_equal(scores1, scores2, decimal=6)


# ── TestScoreAccounts ──────────────────────────────────────────────────────────

class TestScoreAccounts:
    def test_output_count_matches_input(self) -> None:
        feat_df = _make_clear_outlier_feature_df(n_normal=20)
        model = fit_model(feat_df)
        scores = score_accounts(feat_df, model)
        assert len(scores) == len(feat_df)

    def test_schema_valid(self) -> None:
        """All returned objects parse as AnomalyScore without validation error."""
        feat_df = _make_clear_outlier_feature_df(n_normal=20)
        model = fit_model(feat_df)
        scores = score_accounts(feat_df, model)
        for s in scores:
            assert isinstance(s, AnomalyScore)

    def test_score_is_python_float(self) -> None:
        feat_df = _make_clear_outlier_feature_df(n_normal=5)
        model = fit_model(feat_df)
        for s in score_accounts(feat_df, model):
            assert type(s.score) is float, f"score is {type(s.score)}, expected float"

    def test_is_flagged_is_python_bool(self) -> None:
        feat_df = _make_clear_outlier_feature_df(n_normal=5)
        model = fit_model(feat_df)
        for s in score_accounts(feat_df, model):
            assert type(s.is_flagged) is bool, (
                f"is_flagged is {type(s.is_flagged)}, expected bool"
            )

    def test_percentile_in_unit_interval(self) -> None:
        feat_df = _make_clear_outlier_feature_df(n_normal=20)
        model = fit_model(feat_df)
        for s in score_accounts(feat_df, model):
            assert 0.0 <= s.percentile <= 1.0, (
                f"percentile {s.percentile} out of [0,1] for {s.account_id}"
            )

    def test_most_anomalous_has_highest_percentile(self) -> None:
        """Extreme outlier account gets the highest anomaly percentile."""
        feat_df = _make_clear_outlier_feature_df(n_normal=20)
        model = fit_model(feat_df)
        scores = score_accounts(feat_df, model)
        outlier = next(s for s in scores if s.account_id == "OUTLIER")
        others = [s for s in scores if s.account_id != "OUTLIER"]
        assert all(outlier.percentile >= s.percentile for s in others)

    def test_most_anomalous_has_lowest_score(self) -> None:
        """Extreme outlier has the most negative (lowest) raw score."""
        feat_df = _make_clear_outlier_feature_df(n_normal=20)
        model = fit_model(feat_df)
        scores = score_accounts(feat_df, model)
        outlier = next(s for s in scores if s.account_id == "OUTLIER")
        others = [s for s in scores if s.account_id != "OUTLIER"]
        assert all(outlier.score <= s.score for s in others)

    def test_score_and_percentile_directions_consistent(self) -> None:
        """Lower score → higher anomaly percentile (no direction inversion)."""
        feat_df = _make_clear_outlier_feature_df(n_normal=20)
        model = fit_model(feat_df)
        scores = score_accounts(feat_df, model)
        raw = np.array([s.score for s in scores])
        pct = np.array([s.percentile for s in scores])
        # Pearson correlation between -raw_score and percentile should be > 0
        corr = np.corrcoef(-raw, pct)[0, 1]
        assert corr > 0.9, (
            f"score and percentile directions are inconsistent (corr={corr:.3f})"
        )

    def test_tie_same_percentile(self) -> None:
        """Accounts with identical features → identical raw scores → identical percentile."""
        n = 20
        values = [1.0] * n
        values[0] = 1000.0  # outlier
        # Accounts at indices 5 and 6 are identical (both = 1.0)
        # so they'll have the same IF score and must get the same percentile
        feat_df = pl.DataFrame({
            "account_id": [f"ACC{i:02d}" for i in range(n)],
            **{col: values for col in FEATURE_COLS},
        }).with_columns([pl.col(c).cast(pl.Float32) for c in FEATURE_COLS])

        model = fit_model(feat_df)
        scores_list = score_accounts(feat_df, model)
        score_by_id = {s.account_id: s for s in scores_list}

        # All non-outlier accounts have identical features → identical IF scores
        normal_scores = [score_by_id[f"ACC{i:02d}"] for i in range(1, n)]
        percentiles = [s.percentile for s in normal_scores]
        assert all(p == pytest.approx(percentiles[0]) for p in percentiles), (
            f"Tied accounts have different percentiles: {sorted(set(percentiles))}"
        )

    def test_determinism(self) -> None:
        """Two calls with same inputs produce bitwise-identical scores."""
        feat_df = _make_clear_outlier_feature_df(n_normal=20)
        model = fit_model(feat_df)
        scores1 = score_accounts(feat_df, model)
        scores2 = score_accounts(feat_df, model)
        for s1, s2 in zip(scores1, scores2):
            assert s1.account_id == s2.account_id
            assert s1.score == s2.score
            assert s1.percentile == s2.percentile

    def test_excluded_features_in_every_score_object(self) -> None:
        feat_df = _make_clear_outlier_feature_df(n_normal=5)
        model = fit_model(feat_df)
        for s in score_accounts(feat_df, model):
            assert "is_laundering" in s.excluded_features
            assert "pattern_label" in s.excluded_features
            assert len(s.excluded_features) == len(EXCLUDED_FEATURES)

    def test_flagged_accounts_at_top_percentile(self) -> None:
        """is_flagged iff percentile >= ANOMALY_FLAGGING_PERCENTILE."""
        feat_df = _make_clear_outlier_feature_df(n_normal=20)
        model = fit_model(feat_df)
        for s in score_accounts(feat_df, model):
            expected = s.percentile >= ANOMALY_FLAGGING_PERCENTILE
            assert s.is_flagged == expected, (
                f"{s.account_id}: is_flagged={s.is_flagged} "
                f"but percentile={s.percentile:.4f} vs threshold={ANOMALY_FLAGGING_PERCENTILE}"
            )

    def test_exclusion_logged(self, tmp_path: Path) -> None:
        """Running scorer with log_path creates a log file containing excluded feature names."""
        log_path = tmp_path / "excluded.log"
        feat_df = _make_clear_outlier_feature_df(n_normal=5)
        model = fit_model(feat_df)
        score_accounts(feat_df, model, log_path=log_path)

        assert log_path.exists(), "Log file was not created"
        content = log_path.read_text()
        assert "EXCLUDED_FEATURE: is_laundering" in content
        assert "EXCLUDED_FEATURE: pattern_label" in content
        for feat in EXCLUDED_FEATURES:
            assert feat in content, f"Excluded feature {feat!r} not found in log"


# ── TestGetScore ───────────────────────────────────────────────────────────────

class TestGetScore:
    def test_found(self) -> None:
        feat_df = _make_clear_outlier_feature_df(n_normal=5)
        model = fit_model(feat_df)
        scores = score_accounts(feat_df, model)
        result = get_score("OUTLIER", scores)
        assert result is not None
        assert result.account_id == "OUTLIER"

    def test_missing_returns_none(self) -> None:
        feat_df = _make_clear_outlier_feature_df(n_normal=5)
        model = fit_model(feat_df)
        scores = score_accounts(feat_df, model)
        assert get_score("PHANTOM_XYZ", scores) is None

    def test_returns_correct_score(self) -> None:
        feat_df = _make_clear_outlier_feature_df(n_normal=5)
        model = fit_model(feat_df)
        scores = score_accounts(feat_df, model)
        for s in scores:
            found = get_score(s.account_id, scores)
            assert found is not None
            assert found.score == s.score


# ── TestLogExcludedFeatures ────────────────────────────────────────────────────

class TestLogExcludedFeatures:
    def test_creates_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "subdir" / "excluded.log"
        log_excluded_features(["feat_a", "feat_b"], log_path)
        assert log_path.exists()

    def test_content_contains_each_feature(self, tmp_path: Path) -> None:
        log_path = tmp_path / "test.log"
        features = ["is_laundering", "net_flow"]
        log_excluded_features(features, log_path)
        content = log_path.read_text()
        for f in features:
            assert f in content

    def test_appends_on_repeated_calls(self, tmp_path: Path) -> None:
        log_path = tmp_path / "test.log"
        log_excluded_features(["feat_a"], log_path)
        log_excluded_features(["feat_b"], log_path)
        content = log_path.read_text()
        assert "feat_a" in content
        assert "feat_b" in content
        assert content.count("EXCLUDED_FEATURE:") == 2


# ── TestSanitySeparation (unit, synthetic data) ────────────────────────────────

def test_sanity_separation() -> None:
    """
    Laundering-like accounts (fan-out, off-hours, structuring amounts) have
    higher anomaly percentile than legitimate accounts.
    AUC > 0.55 using y_score = -raw_score (IsolationForest: more negative = more anomalous).

    Uses a 10% minority rate (20 laundering in 200 total) so IsolationForest can
    identify the laundering cluster as isolated from the legit majority.  A 50/50
    split would make both clusters equally easy to isolate — defeating the scorer.
    """
    from sklearn.metrics import roc_auc_score

    # 20 laundering accounts out of 200 total (10% minority)
    df, accounts_df, launder_ids = _make_synth_transactions(
        n_legit=180, n_launder=20, seed=42
    )
    feat_df = build_feature_matrix(df, accounts_df)
    model = fit_model(feat_df)
    scores_list = score_accounts(feat_df, model)

    score_map = {s.account_id: s for s in scores_list}
    launder_set = set(launder_ids)

    labeled_ids = [
        s.account_id for s in scores_list
        if s.account_id.startswith("LEGIT_") or s.account_id.startswith("LAUND_")
    ]
    y_true = [1 if aid in launder_set else 0 for aid in labeled_ids]
    # Negate: higher = more anomalous (IsolationForest: lower score = more anomalous)
    y_score = [-score_map[aid].score for aid in labeled_ids]

    mean_launder = float(np.mean(
        [-score_map[aid].score for aid in launder_ids if aid in score_map]
    ))
    mean_legit = float(np.mean(
        [-score_map[aid].score for aid in labeled_ids
         if aid.startswith("LEGIT_") and aid in score_map]
    ))
    auc = roc_auc_score(y_true, y_score)

    assert mean_launder > mean_legit, (
        f"Laundering accounts should be more anomalous (-score higher). "
        f"mean_launder={mean_launder:.4f}, mean_legit={mean_legit:.4f}"
    )
    assert auc > 0.55, (
        f"Sanity AUC={auc:.4f} < 0.55 — model is below-chance on synthetic sample. "
        f"mean_launder={mean_launder:.4f}, mean_legit={mean_legit:.4f}"
    )


def test_no_label_in_features(tiny_transactions, tiny_accounts) -> None:
    """is_laundering column is not present in the feature DataFrame at any pipeline stage."""
    feat_df = build_feature_matrix(tiny_transactions, tiny_accounts)
    assert "is_laundering" not in feat_df.columns


def test_exclusion_logged(tmp_path: Path, tiny_transactions, tiny_accounts) -> None:
    """Running the scorer creates a log entry naming each excluded feature."""
    log_path = tmp_path / "excluded.log"
    feat_df = build_feature_matrix(tiny_transactions, tiny_accounts)
    model = fit_model(feat_df)
    score_accounts(feat_df, model, log_path=log_path)

    assert log_path.exists()
    content = log_path.read_text()
    for feat in EXCLUDED_FEATURES:
        assert feat in content, f"Excluded feature {feat!r} missing from log"


# ── Integration tests ──────────────────────────────────────────────────────────

@pytest.mark.integration
class TestIntegration:
    _TRANS = Path("data/raw/HI-Small_Trans.csv")

    @pytest.fixture(autouse=True)
    def require_data(self) -> None:
        if not self._TRANS.exists():
            pytest.skip(f"Transaction CSV not found: {self._TRANS}")

    @pytest.fixture
    def real_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        from aml_copilot.step0_scaffold.data_loader import derive_accounts, load_transactions
        df = load_transactions(self._TRANS)
        accounts = derive_accounts(df)
        return df, accounts

    def test_all_accounts_scored(
        self, real_data: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """All 515K accounts receive an AnomalyScore."""
        df, accounts = real_data
        feat_df = build_feature_matrix(df, accounts)
        model = fit_model(feat_df)
        scores = score_accounts(feat_df, model)
        assert len(scores) == len(accounts), (
            f"Expected {len(accounts)} scores, got {len(scores)}"
        )

    def test_no_null_or_inf_scores(
        self, real_data: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        df, accounts = real_data
        feat_df = build_feature_matrix(df, accounts)
        model = fit_model(feat_df)
        for s in score_accounts(feat_df, model):
            assert np.isfinite(s.score), f"Non-finite score for {s.account_id}"

    def test_window_excludes_is_laundering(
        self, real_data: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        df, accounts = real_data
        feat_df = build_feature_matrix(df, accounts)
        assert "is_laundering" not in feat_df.columns

    def test_sanity_integration_auc(
        self, real_data: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """
        AUC > 0.55 on a 200-account labeled sample from HI-Small.
        y_score = -raw_score so that higher = more anomalous.
        """
        from sklearn.metrics import roc_auc_score

        df, accounts = real_data
        feat_df = build_feature_matrix(df, accounts)
        model = fit_model(feat_df)
        scores_list = score_accounts(feat_df, model)
        score_map = {s.account_id: s for s in scores_list}

        # Identify laundering accounts (appear in at least one is_laundering=1 row)
        launder_accts = set(
            df.filter(pl.col("is_laundering") == 1)["from_account"].to_list()
        )
        legit_accts = set(accounts["account_id"].to_list()) - launder_accts

        rng = np.random.default_rng(42)
        n = 100
        sample_launder = rng.choice(list(launder_accts & score_map.keys()), size=n, replace=False)
        sample_legit = rng.choice(list(legit_accts & score_map.keys()), size=n, replace=False)
        labeled = list(sample_launder) + list(sample_legit)
        y_true = [1] * n + [0] * n
        y_score = [-score_map[aid].score for aid in labeled]

        mean_launder = float(np.mean([-score_map[aid].score for aid in sample_launder]))
        mean_legit = float(np.mean([-score_map[aid].score for aid in sample_legit]))
        auc = roc_auc_score(y_true, y_score)

        print(
            f"\nIntegration sanity: AUC={auc:.4f} | "
            f"mean_launder(-score)={mean_launder:.4f} | "
            f"mean_legit(-score)={mean_legit:.4f}"
        )
        assert auc > 0.55, f"Integration sanity AUC={auc:.4f} < 0.55"
