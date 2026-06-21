"""
Assertion-based validation for IBM AMLSim HI-Small dataset (Step 0 DoD gate).

These checks exist to catch silent dataset substitution. The three numbers below
are the fingerprint of HI-Small — not HI-Medium (32 M rows / 2.08 M accounts)
and not a truncated download.

Run via run_all_validations(); raises AssertionError on any failure with a
human-readable message identifying the specific check that failed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

# ── Dataset fingerprint ──────────────────────────────────────────────────────

EXPECTED_ROW_COUNT: int = 5_078_345       # exact; no tolerance
EXPECTED_ACCOUNT_COUNT: int = 515_080     # union of from_account + to_account
ACCOUNT_COUNT_TOLERANCE: int = 100

EXPECTED_LAUNDERING_RATIO: float = 0.001
LAUNDERING_RATIO_TOLERANCE: float = 0.0002  # accepted band: [0.0008, 0.0012]

REQUIRED_COLUMNS: list[str] = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]

_KEY_COLUMNS: list[str] = ["from_account", "to_account", "amount_received"]


# ── Result type ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    check: str
    expected: Any
    actual: Any
    message: str

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.check}: {self.message}"


# ── Individual checks ────────────────────────────────────────────────────────

def validate_schema(df: pl.DataFrame) -> ValidationResult:
    """All REQUIRED_COLUMNS must be present in df."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    passed = len(missing) == 0
    return ValidationResult(
        passed=passed,
        check="schema",
        expected=REQUIRED_COLUMNS,
        actual=df.columns,
        message=(
            "all required columns present"
            if passed
            else f"missing columns: {missing}"
        ),
    )


def validate_no_nulls_in_keys(df: pl.DataFrame) -> ValidationResult:
    """from_account, to_account, and amount_received must have zero nulls."""
    present = [c for c in _KEY_COLUMNS if c in df.columns]
    null_counts: dict[str, int] = {c: df[c].null_count() for c in present}
    violations = {k: v for k, v in null_counts.items() if v > 0}
    passed = len(violations) == 0
    return ValidationResult(
        passed=passed,
        check="no_nulls_in_keys",
        expected=0,
        actual=violations,
        message=(
            "no nulls in key columns"
            if passed
            else f"null values found: {violations}"
        ),
    )


def validate_null_counts(df: pl.DataFrame) -> ValidationResult:
    """No column in df should have any null values after loading."""
    null_counts: dict[str, int] = {
        col: count
        for col in df.columns
        if (count := df[col].null_count()) > 0
    }
    passed = len(null_counts) == 0
    return ValidationResult(
        passed=passed,
        check="null_counts",
        expected=0,
        actual=null_counts,
        message=(
            "no nulls across all columns"
            if passed
            else f"null values found in {len(null_counts)} column(s): {null_counts}"
        ),
    )


def validate_row_count(df: pl.DataFrame) -> ValidationResult:
    """Row count must be exactly EXPECTED_ROW_COUNT (no tolerance — wrong file = wrong number)."""
    actual = len(df)
    passed = actual == EXPECTED_ROW_COUNT
    return ValidationResult(
        passed=passed,
        check="row_count",
        expected=EXPECTED_ROW_COUNT,
        actual=actual,
        message=(
            f"{actual:,} rows — OK"
            if passed
            else (
                f"got {actual:,} rows, expected exactly {EXPECTED_ROW_COUNT:,}. "
                "HI-Medium has ~32 M rows; a truncated download would be less."
            )
        ),
    )


def validate_account_count(accounts: pl.DataFrame) -> ValidationResult:
    """
    Unique account count (len of derive_accounts result) must be within
    EXPECTED_ACCOUNT_COUNT ± ACCOUNT_COUNT_TOLERANCE.

    Raises
    ------
    ValueError
        If accounts is not a single-column DataFrame with column "account_id".
        This guards against accidentally passing the raw transactions DataFrame.
    """
    if accounts.columns != ["account_id"]:
        raise ValueError(
            f"validate_account_count expects a single-column DataFrame with column "
            f"'account_id', got columns: {accounts.columns}. "
            f"Call derive_accounts(df) first."
        )
    actual = len(accounts)
    low = EXPECTED_ACCOUNT_COUNT - ACCOUNT_COUNT_TOLERANCE
    high = EXPECTED_ACCOUNT_COUNT + ACCOUNT_COUNT_TOLERANCE
    passed = low <= actual <= high
    return ValidationResult(
        passed=passed,
        check="account_count",
        expected=f"{EXPECTED_ACCOUNT_COUNT:,} ± {ACCOUNT_COUNT_TOLERANCE}",
        actual=actual,
        message=(
            f"{actual:,} unique accounts — OK"
            if passed
            else (
                f"got {actual:,} unique accounts, expected {low:,}–{high:,}. "
                "HI-Medium has ~2.08 M accounts."
            )
        ),
    )


def validate_laundering_ratio(df: pl.DataFrame) -> ValidationResult:
    """is_laundering mean must be within EXPECTED_LAUNDERING_RATIO ± LAUNDERING_RATIO_TOLERANCE."""
    if "is_laundering" not in df.columns:
        return ValidationResult(
            passed=False,
            check="laundering_ratio",
            expected=f"{EXPECTED_LAUNDERING_RATIO:.4f} ± {LAUNDERING_RATIO_TOLERANCE:.4f}",
            actual="column absent",
            message="is_laundering column missing — cannot compute ratio",
        )
    ratio: float = df["is_laundering"].cast(pl.Float64).mean()  # type: ignore[assignment]
    low = EXPECTED_LAUNDERING_RATIO - LAUNDERING_RATIO_TOLERANCE
    high = EXPECTED_LAUNDERING_RATIO + LAUNDERING_RATIO_TOLERANCE
    passed = low <= ratio <= high
    return ValidationResult(
        passed=passed,
        check="laundering_ratio",
        expected=f"{EXPECTED_LAUNDERING_RATIO:.4f} ± {LAUNDERING_RATIO_TOLERANCE:.4f}",
        actual=f"{ratio:.6f}",
        message=(
            f"laundering rate {ratio:.4%} — OK"
            if passed
            else f"laundering rate {ratio:.4%} outside expected band [{low:.4%}, {high:.4%}]"
        ),
    )


def validate_is_laundering_values(df: pl.DataFrame) -> ValidationResult:
    """is_laundering column must contain only 0 and 1. Uses a single unique-values pass."""
    if "is_laundering" not in df.columns:
        return ValidationResult(
            passed=False,
            check="is_laundering_values",
            expected={0, 1},
            actual="column absent",
            message="is_laundering column missing",
        )
    unique_vals: set[int] = set(df["is_laundering"].unique().cast(pl.Int64).to_list())
    bad_values = sorted(unique_vals - {0, 1})
    passed = len(bad_values) == 0
    return ValidationResult(
        passed=passed,
        check="is_laundering_values",
        expected={0, 1},
        actual=unique_vals if passed else set(bad_values),
        message=(
            "only 0/1 values — OK"
            if passed
            else f"unexpected values in is_laundering: {bad_values}"
        ),
    )


# ── DoD gate ─────────────────────────────────────────────────────────────────

def run_all_validations(
    df: pl.DataFrame,
    accounts: pl.DataFrame,
) -> list[ValidationResult]:
    """
    Run all 7 validation checks in order.

    Raises
    ------
    AssertionError
        On the first batch of failures, with a message listing every check
        that failed. Individual passing checks are printed to stdout for
        visibility during data loading.
    ValueError
        If accounts does not have exactly the column schema produced by
        derive_accounts() (propagated from validate_account_count).
    """
    results = [
        validate_schema(df),
        validate_no_nulls_in_keys(df),
        validate_null_counts(df),
        validate_is_laundering_values(df),
        validate_row_count(df),
        validate_account_count(accounts),
        validate_laundering_ratio(df),
    ]

    for r in results:
        print(r)

    failures = [r for r in results if not r.passed]
    if failures:
        lines = "\n".join(f"  {r}" for r in failures)
        raise AssertionError(
            f"Step 0 DoD gate failed ({len(failures)} check(s)):\n{lines}"
        )

    return results
