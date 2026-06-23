"""
Integration tests for Step 0 — require the actual HI-Small dataset on disk.

These tests are skipped automatically when the data file is absent.
They are marked `integration` so they can be excluded from fast CI runs:

    pytest -m "not integration"   # skip these
    pytest -m integration         # run only these

Set TRANSACTIONS_PATH (and optionally PATTERNS_PATH) env vars to override
the default paths, or place the files under data/raw/ relative to the project root.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import polars as pl
import pytest

from aml_copilot.step0_scaffold.data_loader import (
    COLUMN_NAMES,
    derive_accounts,
    load_patterns,
    load_transactions,
)
from aml_copilot.step0_scaffold.validation import (
    EXPECTED_ACCOUNT_COUNT,
    EXPECTED_LAUNDERING_RATIO,
    EXPECTED_ROW_COUNT,
    LAUNDERING_RATIO_TOLERANCE,
    run_all_validations,
    validate_account_count,
    validate_laundering_ratio,
    validate_row_count,
)

pytestmark = pytest.mark.integration

# ── Path resolution ──────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent


def _resolve(env_var: str, default_relative: str) -> Path:
    return Path(os.environ.get(env_var, _PROJECT_ROOT / default_relative))


TRANSACTIONS_PATH = _resolve("TRANSACTIONS_PATH", "data/raw/HI-Small_Trans.csv")
PATTERNS_PATH = _resolve("PATTERNS_PATH", "data/raw/HI-Small_Patterns.txt")


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def transactions() -> pl.DataFrame:
    if not TRANSACTIONS_PATH.exists():
        pytest.skip(
            f"HI-Small_Trans.csv not found at {TRANSACTIONS_PATH}. "
            "Set the TRANSACTIONS_PATH environment variable or place the file under data/raw/."
        )
    return load_transactions(TRANSACTIONS_PATH)


@pytest.fixture(scope="module")
def accounts(transactions: pl.DataFrame) -> pl.DataFrame:
    return derive_accounts(transactions)


# ── Load tests ───────────────────────────────────────────────────────────────

class TestLoadTransactionsIntegration:
    def test_returns_polars_dataframe(self, transactions: pl.DataFrame) -> None:
        assert isinstance(transactions, pl.DataFrame)

    def test_has_all_canonical_columns(self, transactions: pl.DataFrame) -> None:
        assert transactions.columns == COLUMN_NAMES

    def test_timestamp_dtype_is_datetime(self, transactions: pl.DataFrame) -> None:
        assert transactions.schema["timestamp"] == pl.Datetime

    def test_is_laundering_dtype_is_int8(self, transactions: pl.DataFrame) -> None:
        assert transactions.schema["is_laundering"] == pl.Int8

    def test_amount_received_dtype_is_float64(self, transactions: pl.DataFrame) -> None:
        assert transactions.schema["amount_received"] == pl.Float64

    def test_load_performance_under_60s(self) -> None:
        """Loading ~5M rows should complete in under 60 seconds on any modern machine."""
        if not TRANSACTIONS_PATH.exists():
            pytest.skip("data file not found")
        start = time.perf_counter()
        load_transactions(TRANSACTIONS_PATH)
        elapsed = time.perf_counter() - start
        assert elapsed < 60.0, f"load_transactions took {elapsed:.1f}s — expected < 60s"


# ── Row count (DoD assertion) ─────────────────────────────────────────────────

class TestRowCountIntegration:
    def test_exact_row_count(self, transactions: pl.DataFrame) -> None:
        result = validate_row_count(transactions)
        assert result.passed, result.message

    def test_row_count_value(self, transactions: pl.DataFrame) -> None:
        assert len(transactions) == EXPECTED_ROW_COUNT

    def test_no_empty_dataframe(self, transactions: pl.DataFrame) -> None:
        assert len(transactions) > 0


# ── Account count (DoD assertion) ────────────────────────────────────────────

class TestAccountCountIntegration:
    def test_account_count_within_tolerance(self, accounts: pl.DataFrame) -> None:
        result = validate_account_count(accounts)
        assert result.passed, result.message

    def test_account_count_value(self, accounts: pl.DataFrame) -> None:
        low = EXPECTED_ACCOUNT_COUNT - 100
        high = EXPECTED_ACCOUNT_COUNT + 100
        assert low <= len(accounts) <= high, (
            f"Account count {len(accounts):,} outside expected [{low:,}, {high:,}]"
        )

    def test_account_ids_are_unique(self, accounts: pl.DataFrame) -> None:
        assert accounts["account_id"].n_unique() == len(accounts)

    def test_account_column_name(self, accounts: pl.DataFrame) -> None:
        assert accounts.columns == ["account_id"]

    def test_from_accounts_subset_of_result(self, transactions: pl.DataFrame, accounts: pl.DataFrame) -> None:
        account_set = set(accounts["account_id"].to_list())
        from_set = set(transactions["from_account"].unique().to_list())
        assert from_set <= account_set

    def test_to_accounts_subset_of_result(self, transactions: pl.DataFrame, accounts: pl.DataFrame) -> None:
        account_set = set(accounts["account_id"].to_list())
        to_set = set(transactions["to_account"].unique().to_list())
        assert to_set <= account_set


# ── Laundering ratio (DoD assertion) ─────────────────────────────────────────

class TestLaunderingRatioIntegration:
    def test_ratio_within_tolerance(self, transactions: pl.DataFrame) -> None:
        result = validate_laundering_ratio(transactions)
        assert result.passed, result.message

    def test_ratio_approximately_0_1_percent(self, transactions: pl.DataFrame) -> None:
        ratio: float = transactions["is_laundering"].mean()  # type: ignore[assignment]
        low = EXPECTED_LAUNDERING_RATIO - LAUNDERING_RATIO_TOLERANCE
        high = EXPECTED_LAUNDERING_RATIO + LAUNDERING_RATIO_TOLERANCE
        assert low <= ratio <= high, f"Laundering ratio {ratio:.6f} outside [{low}, {high}]"

    def test_laundering_count_nonzero(self, transactions: pl.DataFrame) -> None:
        n_laundering = transactions["is_laundering"].sum()
        assert n_laundering > 0

    def test_legit_count_dominates(self, transactions: pl.DataFrame) -> None:
        n_laundering = transactions["is_laundering"].sum()
        assert n_laundering < len(transactions) * 0.01  # < 1%


# ── Data quality ─────────────────────────────────────────────────────────────

class TestDataQualityIntegration:
    def test_no_null_from_accounts(self, transactions: pl.DataFrame) -> None:
        assert transactions["from_account"].null_count() == 0

    def test_no_null_to_accounts(self, transactions: pl.DataFrame) -> None:
        assert transactions["to_account"].null_count() == 0

    def test_no_null_amounts(self, transactions: pl.DataFrame) -> None:
        assert transactions["amount_received"].null_count() == 0

    def test_is_laundering_only_0_and_1(self, transactions: pl.DataFrame) -> None:
        unique_values = set(transactions["is_laundering"].unique().to_list())
        assert unique_values <= {0, 1}

    def test_amounts_are_positive(self, transactions: pl.DataFrame) -> None:
        assert (transactions["amount_received"] > 0).all()

    def test_timestamps_are_ordered_range(self, transactions: pl.DataFrame) -> None:
        """Timestamps should span a multi-month range consistent with AMLSim generation."""
        min_ts = transactions["timestamp"].min()
        max_ts = transactions["timestamp"].max()
        assert min_ts is not None and max_ts is not None
        assert max_ts > min_ts


# ── Full DoD gate ─────────────────────────────────────────────────────────────

class TestDoDGateIntegration:
    def test_run_all_validations_passes(
        self, transactions: pl.DataFrame, accounts: pl.DataFrame
    ) -> None:
        """The full DoD gate must pass without raising AssertionError."""
        results = run_all_validations(transactions, accounts)
        assert all(r.passed for r in results)

    def test_dod_gate_returns_seven_results(
        self, transactions: pl.DataFrame, accounts: pl.DataFrame
    ) -> None:
        results = run_all_validations(transactions, accounts)
        assert len(results) == 7


# ── Patterns file (optional) ──────────────────────────────────────────────────

class TestLoadPatternsIntegration:
    def test_load_patterns_returns_dict(self) -> None:
        if not PATTERNS_PATH.exists():
            pytest.skip(f"Patterns file not found at {PATTERNS_PATH}")
        result = load_patterns(PATTERNS_PATH)
        assert isinstance(result, dict)

    def test_patterns_have_string_keys(self) -> None:
        if not PATTERNS_PATH.exists():
            pytest.skip(f"Patterns file not found at {PATTERNS_PATH}")
        result = load_patterns(PATTERNS_PATH)
        assert all(isinstance(k, str) for k in result.keys())

    def test_patterns_have_list_values(self) -> None:
        if not PATTERNS_PATH.exists():
            pytest.skip(f"Patterns file not found at {PATTERNS_PATH}")
        result = load_patterns(PATTERNS_PATH)
        assert all(isinstance(v, list) for v in result.values())

    def test_patterns_nonempty(self) -> None:
        if not PATTERNS_PATH.exists():
            pytest.skip(f"Patterns file not found at {PATTERNS_PATH}")
        result = load_patterns(PATTERNS_PATH)
        assert len(result) > 0
