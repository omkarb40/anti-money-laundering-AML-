from __future__ import annotations

import polars as pl

EXPECTED_ROW_COUNT: int = 5_078_345
EXPECTED_ACCOUNT_COUNT: int = 515_080
ACCOUNT_COUNT_TOLERANCE: int = 100
EXPECTED_LAUNDERING_RATIO: float = 0.001
LAUNDERING_RATIO_TOLERANCE: float = 0.0002

REQUIRED_COLUMNS: dict[str, type] = {
    "Timestamp": pl.Utf8,
    "From Account ID": pl.Utf8,
    "To Account ID": pl.Utf8,
    "Amount (Received)": pl.Float64,
    "Payment Type": pl.Utf8,
    "Is Laundering": pl.Int64,
}


def assert_transaction_count(df: pl.DataFrame) -> None:
    """Raise AssertionError if row count != EXPECTED_ROW_COUNT."""
    ...


def assert_account_count(accounts: pl.DataFrame) -> None:
    """Raise AssertionError if account count is outside tolerance."""
    ...


def assert_laundering_ratio(df: pl.DataFrame) -> None:
    """Raise AssertionError if Is Laundering ratio is outside tolerance."""
    ...


def assert_schema(df: pl.DataFrame) -> None:
    """Raise AssertionError if any required column is missing or mistyped."""
    ...


def run_all_assertions(df: pl.DataFrame, accounts: pl.DataFrame) -> None:
    """Run all four assertions in sequence. Called as the Step 0 DoD gate."""
    ...
