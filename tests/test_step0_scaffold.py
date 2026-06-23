"""
Unit tests for Step 0: data_loader and validation.

All tests use small synthetic DataFrames or temp files — no real HI-Small data.
Constants are monkeypatched where the real expected values (5M rows, 515K accounts)
would make fixture creation impractical.

For tests against actual HI-Small data, see test_step0_integration.py.
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

import aml_copilot.step0_scaffold.validation as v_mod
from aml_copilot.step0_scaffold.data_loader import (
    COLUMN_NAMES,
    TIMESTAMP_FORMATS,
    _detect_timestamp_format,
    derive_accounts,
    inspect_raw_csv,
    load_patterns,
    load_transactions,
)
from aml_copilot.step0_scaffold.validation import (
    EXPECTED_ACCOUNT_COUNT,
    EXPECTED_LAUNDERING_RATIO,
    EXPECTED_ROW_COUNT,
    ValidationResult,
    validate_account_count,
    validate_is_laundering_values,
    validate_laundering_ratio,
    validate_no_nulls_in_keys,
    validate_null_counts,
    validate_row_count,
    validate_schema,
    run_all_validations,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_CSV_HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,"
    "Amount Received,Receiving Currency,Amount Paid,Payment Currency,"
    "Payment Format,Is Laundering\n"
)

def _make_csv(tmp_path: Path, rows: list[str]) -> Path:
    csv_file = tmp_path / "HI-Small_Trans.csv"
    csv_file.write_text(_CSV_HEADER + "\n".join(rows))
    return csv_file


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def valid_df() -> pl.DataFrame:
    """Minimal schema-valid DataFrame (3 rows, mixed laundering)."""
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2022, 1, 1, 0, 0),
                datetime(2022, 1, 1, 1, 0),
                datetime(2022, 1, 2, 0, 0),
            ],
            "from_bank": ["BankA", "BankA", "BankB"],
            "from_account": ["ACC001", "ACC002", "ACC003"],
            "to_bank": ["BankB", "BankC", "BankA"],
            "to_account": ["ACC002", "ACC003", "ACC001"],
            "amount_received": [500.0, 1000.0, 200.0],
            "receiving_currency": ["USD", "USD", "EUR"],
            "amount_paid": [500.0, 1000.0, 200.0],
            "payment_currency": ["USD", "USD", "EUR"],
            "payment_format": ["Wire", "Credit Card", "Wire"],
            "is_laundering": pl.Series([0, 1, 0], dtype=pl.Int8),
        }
    )


@pytest.fixture
def valid_accounts() -> pl.DataFrame:
    return pl.DataFrame({"account_id": ["ACC001", "ACC002", "ACC003"]})


# ── data_loader: derive_accounts ─────────────────────────────────────────────

class TestDeriveAccounts:
    def test_union_deduplicates(self, valid_df: pl.DataFrame) -> None:
        accounts = derive_accounts(valid_df)
        assert accounts["account_id"].n_unique() == len(accounts)

    def test_all_accounts_present(self, valid_df: pl.DataFrame) -> None:
        accounts = derive_accounts(valid_df)
        account_set = set(accounts["account_id"].to_list())
        for col in ("from_account", "to_account"):
            for val in valid_df[col].to_list():
                assert val in account_set

    def test_output_column_name(self, valid_df: pl.DataFrame) -> None:
        assert derive_accounts(valid_df).columns == ["account_id"]

    def test_single_transaction(self) -> None:
        df = pl.DataFrame({"from_account": ["A"], "to_account": ["B"]})
        assert set(derive_accounts(df)["account_id"].to_list()) == {"A", "B"}

    def test_self_transfer_does_not_duplicate(self) -> None:
        df = pl.DataFrame({"from_account": ["A"], "to_account": ["A"]})
        assert len(derive_accounts(df)) == 1

    def test_result_is_dataframe_not_series(self, valid_df: pl.DataFrame) -> None:
        result = derive_accounts(valid_df)
        assert isinstance(result, pl.DataFrame)


# ── data_loader: load_transactions ───────────────────────────────────────────

class TestLoadTransactions:
    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_transactions(tmp_path / "missing.csv")

    def test_column_names_after_load(self, tmp_path: Path) -> None:
        """Duplicate Account headers are resolved to canonical names."""
        f = _make_csv(tmp_path, ["2022/01/01 00:00,BankA,ACC001,BankB,ACC002,500.0,USD,500.0,USD,Wire,0"])
        assert load_transactions(f).columns == COLUMN_NAMES

    def test_from_account_is_string_dtype(self, tmp_path: Path) -> None:
        """from_account must be String even when values look like integers."""
        f = _make_csv(tmp_path, ["2022/01/01 00:00,BankA,8000001234,BankB,8000005678,500.0,USD,500.0,USD,Wire,0"])
        df = load_transactions(f)
        assert df.schema["from_account"] == pl.String

    def test_to_account_is_string_dtype(self, tmp_path: Path) -> None:
        f = _make_csv(tmp_path, ["2022/01/01 00:00,BankA,1001,BankB,1002,500.0,USD,500.0,USD,Wire,0"])
        df = load_transactions(f)
        assert df.schema["to_account"] == pl.String

    def test_leading_zeros_in_account_id_preserved(self, tmp_path: Path) -> None:
        """Int64 inference would destroy '00012345' → 12345. String must be preserved."""
        f = _make_csv(tmp_path, ["2022/01/01 00:00,BankA,00012345,BankB,00067890,500.0,USD,500.0,USD,Wire,0"])
        df = load_transactions(f)
        assert df["from_account"][0] == "00012345"
        assert df["to_account"][0] == "00067890"

    def test_timestamp_parsed_to_datetime(self, tmp_path: Path) -> None:
        f = _make_csv(tmp_path, ["2022/03/15 14:30,BankA,ACC001,BankB,ACC002,500.0,USD,500.0,USD,Wire,0"])
        assert load_transactions(f).schema["timestamp"] == pl.Datetime

    def test_is_laundering_cast_to_int8(self, tmp_path: Path) -> None:
        f = _make_csv(tmp_path, ["2022/01/01 00:00,BankA,ACC001,BankB,ACC002,500.0,USD,500.0,USD,Wire,1"])
        assert load_transactions(f).schema["is_laundering"] == pl.Int8

    def test_amount_cast_to_float64(self, tmp_path: Path) -> None:
        f = _make_csv(tmp_path, ["2022/01/01 00:00,BankA,ACC001,BankB,ACC002,1234.56,USD,1234.56,USD,Wire,0"])
        df = load_transactions(f)
        assert df.schema["amount_received"] == pl.Float64
        assert df["amount_received"][0] == pytest.approx(1234.56)

    def test_multiple_rows_loaded_correctly(self, tmp_path: Path) -> None:
        rows = [
            f"2022/01/0{i+1} 00:00,BankA,ACC{i:03d},BankB,ACC{i+1:03d},100.0,USD,100.0,USD,Wire,0"
            for i in range(5)
        ]
        f = _make_csv(tmp_path, rows)
        assert len(load_transactions(f)) == 5


# ── data_loader: timestamp detection ─────────────────────────────────────────

class TestTimestampDetection:
    def test_detects_slash_no_seconds(self, tmp_path: Path) -> None:
        f = tmp_path / "t.csv"
        f.write_text("h1,h2\n2022/01/15 14:30,x\n")
        assert _detect_timestamp_format(f) == "%Y/%m/%d %H:%M"

    def test_detects_slash_with_seconds(self, tmp_path: Path) -> None:
        f = tmp_path / "t.csv"
        f.write_text("h1,h2\n2022/01/15 14:30:00,x\n")
        assert _detect_timestamp_format(f) == "%Y/%m/%d %H:%M:%S"

    def test_detects_iso_no_seconds(self, tmp_path: Path) -> None:
        f = tmp_path / "t.csv"
        f.write_text("h1,h2\n2022-01-15 14:30,x\n")
        assert _detect_timestamp_format(f) == "%Y-%m-%d %H:%M"

    def test_detects_iso_with_seconds(self, tmp_path: Path) -> None:
        f = tmp_path / "t.csv"
        f.write_text("h1,h2\n2022-01-15 14:30:45,x\n")
        assert _detect_timestamp_format(f) == "%Y-%m-%d %H:%M:%S"

    def test_raises_on_unknown_format_names_tried_formats(self, tmp_path: Path) -> None:
        f = tmp_path / "t.csv"
        f.write_text("h1,h2\n15-01-2022 14:30,x\n")
        with pytest.raises(ValueError, match="TIMESTAMP_FORMATS"):
            _detect_timestamp_format(f)

    def test_raises_includes_sample_value(self, tmp_path: Path) -> None:
        f = tmp_path / "t.csv"
        f.write_text("h1,h2\n20220115T1430,x\n")
        with pytest.raises(ValueError, match="20220115T1430"):
            _detect_timestamp_format(f)

    def test_raises_on_no_data_rows(self, tmp_path: Path) -> None:
        f = tmp_path / "t.csv"
        f.write_text("header_only\n")
        with pytest.raises(ValueError, match="No data rows"):
            _detect_timestamp_format(f)

    def test_load_transactions_uses_detected_format(self, tmp_path: Path) -> None:
        """load_transactions must not hard-code the format."""
        f = _make_csv(tmp_path, ["2022/03/01 08:00,BankA,ACC1,BankB,ACC2,100.0,USD,100.0,USD,Wire,0"])
        df = load_transactions(f)
        assert df.schema["timestamp"] == pl.Datetime


# ── data_loader: inspect_raw_csv ─────────────────────────────────────────────

class TestInspectRawCsv:
    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            inspect_raw_csv(tmp_path / "missing.csv")

    def test_prints_all_column_names(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        f = tmp_path / "t.csv"
        f.write_text("Timestamp,From Bank,Account,To Bank,Amount\n2022/01/01 00:00,A,1,B,100\n")
        inspect_raw_csv(f)
        out = capsys.readouterr().out
        assert "Timestamp" in out
        assert "From Bank" in out
        assert "Amount" in out

    def test_shows_first_n_rows(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        rows = [f"2022/01/0{i+1} 00:00,x,y\n" for i in range(5)]
        f = tmp_path / "t.csv"
        f.write_text("H1,H2,H3\n" + "".join(rows))
        inspect_raw_csv(f, n_rows=3)
        out = capsys.readouterr().out
        assert "row 1" in out
        assert "row 3" in out
        assert "row 4" not in out

    def test_shows_duplicate_renaming_for_duplicate_headers(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        f = tmp_path / "t.csv"
        f.write_text("Timestamp,Account,Account\n2022/01/01 00:00,A,B\n")
        inspect_raw_csv(f)
        out = capsys.readouterr().out
        assert "duplicated" in out

    def test_no_duplicate_section_when_no_duplicates(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        f = tmp_path / "t.csv"
        f.write_text("Col1,Col2,Col3\nA,B,C\n")
        inspect_raw_csv(f)
        out = capsys.readouterr().out
        assert "duplicated" not in out


# ── data_loader: load_patterns ───────────────────────────────────────────────

class TestLoadPatterns:
    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_patterns(tmp_path / "missing.txt")

    def test_basic_parsing(self, tmp_path: Path) -> None:
        f = tmp_path / "patterns.txt"
        f.write_text("fan_out ACC001 ACC002 ACC003\nfan_in ACC004 ACC005\n")
        result = load_patterns(f)
        assert result["fan_out"] == ["ACC001", "ACC002", "ACC003"]
        assert result["fan_in"] == ["ACC004", "ACC005"]

    def test_comment_lines_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "patterns.txt"
        f.write_text("# this is a comment\nfan_out ACC001 ACC002\n")
        result = load_patterns(f)
        assert "#" not in result
        assert "fan_out" in result

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "patterns.txt"
        f.write_text("\n\nfan_out ACC001\n\n")
        assert len(load_patterns(f)) == 1

    def test_comma_separated_accepted(self, tmp_path: Path) -> None:
        f = tmp_path / "patterns.txt"
        f.write_text("cycle,ACC001,ACC002,ACC003\n")
        result = load_patterns(f)
        assert "cycle" in result
        assert "ACC001" in result["cycle"]

    def test_same_typology_multiple_lines_merged(self, tmp_path: Path) -> None:
        f = tmp_path / "patterns.txt"
        f.write_text("fan_out ACC001 ACC002\nfan_out ACC003\n")
        assert set(load_patterns(f)["fan_out"]) == {"ACC001", "ACC002", "ACC003"}

    def test_single_token_line_warns(self, tmp_path: Path) -> None:
        f = tmp_path / "patterns.txt"
        f.write_text("orphan_token\n")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = load_patterns(f)
        assert len(result) == 0
        assert any("skipped" in str(w.message).lower() for w in caught)

    def test_empty_file_returns_empty_dict(self, tmp_path: Path) -> None:
        f = tmp_path / "patterns.txt"
        f.write_text("")
        assert load_patterns(f) == {}


# ── validation: validate_schema ──────────────────────────────────────────────

class TestValidateSchema:
    def test_passes_on_valid_df(self, valid_df: pl.DataFrame) -> None:
        assert validate_schema(valid_df).passed

    def test_fails_when_column_missing(self, valid_df: pl.DataFrame) -> None:
        result = validate_schema(valid_df.drop("from_account"))
        assert not result.passed
        assert "from_account" in result.message

    def test_fails_lists_all_missing(self, valid_df: pl.DataFrame) -> None:
        result = validate_schema(valid_df.drop(["from_account", "to_account"]))
        assert not result.passed
        assert "from_account" in result.message
        assert "to_account" in result.message

    def test_extra_columns_do_not_fail(self, valid_df: pl.DataFrame) -> None:
        df = valid_df.with_columns(pl.lit("extra").alias("extra_col"))
        assert validate_schema(df).passed

    def test_result_is_validation_result(self, valid_df: pl.DataFrame) -> None:
        assert isinstance(validate_schema(valid_df), ValidationResult)


# ── validation: validate_no_nulls_in_keys ───────────────────────────────────

class TestValidateNoNullsInKeys:
    def test_passes_with_no_nulls(self, valid_df: pl.DataFrame) -> None:
        assert validate_no_nulls_in_keys(valid_df).passed

    def test_fails_on_null_from_account(self, valid_df: pl.DataFrame) -> None:
        df = valid_df.with_columns(
            pl.when(pl.col("from_account") == "ACC001")
            .then(None)
            .otherwise(pl.col("from_account"))
            .alias("from_account")
        )
        result = validate_no_nulls_in_keys(df)
        assert not result.passed
        assert "from_account" in str(result.actual)

    def test_fails_on_null_amount(self, valid_df: pl.DataFrame) -> None:
        df = valid_df.with_columns(
            pl.when(pl.col("amount_received") == 500.0)
            .then(None)
            .otherwise(pl.col("amount_received"))
            .cast(pl.Float64)
            .alias("amount_received")
        )
        assert not validate_no_nulls_in_keys(df).passed

    def test_skips_missing_columns_gracefully(self) -> None:
        df = pl.DataFrame({"other_col": [1, 2, 3]})
        assert validate_no_nulls_in_keys(df).passed


# ── validation: validate_null_counts ─────────────────────────────────────────

class TestValidateNullCounts:
    def test_passes_when_no_nulls(self, valid_df: pl.DataFrame) -> None:
        assert validate_null_counts(valid_df).passed

    def test_fails_on_any_null_in_any_column(self, valid_df: pl.DataFrame) -> None:
        df = valid_df.with_columns(
            pl.when(pl.col("from_bank") == "BankA")
            .then(None)
            .otherwise(pl.col("from_bank"))
            .alias("from_bank")
        )
        result = validate_null_counts(df)
        assert not result.passed
        assert "from_bank" in str(result.actual)

    def test_reports_all_columns_with_nulls(self, valid_df: pl.DataFrame) -> None:
        df = valid_df.with_columns(
            pl.when(pl.col("from_bank") == "BankA").then(None).otherwise(pl.col("from_bank")).alias("from_bank"),
            pl.when(pl.col("to_bank") == "BankB").then(None).otherwise(pl.col("to_bank")).alias("to_bank"),
        )
        result = validate_null_counts(df)
        assert not result.passed
        assert "from_bank" in str(result.actual)
        assert "to_bank" in str(result.actual)

    def test_null_count_reflects_number_of_null_rows(self, valid_df: pl.DataFrame) -> None:
        df = valid_df.with_columns(
            pl.when(pl.col("from_bank") == "BankA")
            .then(None)
            .otherwise(pl.col("from_bank"))
            .alias("from_bank")
        )
        result = validate_null_counts(df)
        assert result.actual["from_bank"] == 2  # "BankA" appears in rows 0 and 1

    def test_passes_on_empty_dataframe(self) -> None:
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.String)})
        assert validate_null_counts(df).passed


# ── validation: validate_row_count ───────────────────────────────────────────

class TestValidateRowCount:
    def test_passes_on_exact_count(
        self, monkeypatch: pytest.MonkeyPatch, valid_df: pl.DataFrame
    ) -> None:
        monkeypatch.setattr(v_mod, "EXPECTED_ROW_COUNT", 3)
        assert validate_row_count(valid_df).passed

    def test_fails_when_one_too_many(
        self, monkeypatch: pytest.MonkeyPatch, valid_df: pl.DataFrame
    ) -> None:
        monkeypatch.setattr(v_mod, "EXPECTED_ROW_COUNT", 2)
        result = validate_row_count(valid_df)
        assert not result.passed
        assert result.actual == 3

    def test_fails_when_one_too_few(
        self, monkeypatch: pytest.MonkeyPatch, valid_df: pl.DataFrame
    ) -> None:
        monkeypatch.setattr(v_mod, "EXPECTED_ROW_COUNT", 4)
        assert not validate_row_count(valid_df).passed

    def test_message_contains_actual_count(
        self, monkeypatch: pytest.MonkeyPatch, valid_df: pl.DataFrame
    ) -> None:
        monkeypatch.setattr(v_mod, "EXPECTED_ROW_COUNT", 99)
        result = validate_row_count(valid_df)
        assert str(result.actual) in result.message

    def test_result_actual_reflects_df_len(self, valid_df: pl.DataFrame) -> None:
        assert validate_row_count(valid_df).actual == len(valid_df)


# ── validation: validate_account_count ──────────────────────────────────────

class TestValidateAccountCount:
    def test_passes_at_exact_count(
        self, monkeypatch: pytest.MonkeyPatch, valid_accounts: pl.DataFrame
    ) -> None:
        monkeypatch.setattr(v_mod, "EXPECTED_ACCOUNT_COUNT", 3)
        monkeypatch.setattr(v_mod, "ACCOUNT_COUNT_TOLERANCE", 0)
        assert validate_account_count(valid_accounts).passed

    def test_passes_within_tolerance(
        self, monkeypatch: pytest.MonkeyPatch, valid_accounts: pl.DataFrame
    ) -> None:
        monkeypatch.setattr(v_mod, "EXPECTED_ACCOUNT_COUNT", 5)
        monkeypatch.setattr(v_mod, "ACCOUNT_COUNT_TOLERANCE", 3)
        assert validate_account_count(valid_accounts).passed

    def test_fails_outside_tolerance(
        self, monkeypatch: pytest.MonkeyPatch, valid_accounts: pl.DataFrame
    ) -> None:
        monkeypatch.setattr(v_mod, "EXPECTED_ACCOUNT_COUNT", 10)
        monkeypatch.setattr(v_mod, "ACCOUNT_COUNT_TOLERANCE", 1)
        assert not validate_account_count(valid_accounts).passed

    def test_boundary_at_tolerance_edge_passes(
        self, monkeypatch: pytest.MonkeyPatch, valid_accounts: pl.DataFrame
    ) -> None:
        monkeypatch.setattr(v_mod, "EXPECTED_ACCOUNT_COUNT", 6)
        monkeypatch.setattr(v_mod, "ACCOUNT_COUNT_TOLERANCE", 3)
        assert validate_account_count(valid_accounts).passed

    def test_boundary_one_beyond_fails(
        self, monkeypatch: pytest.MonkeyPatch, valid_accounts: pl.DataFrame
    ) -> None:
        monkeypatch.setattr(v_mod, "EXPECTED_ACCOUNT_COUNT", 7)
        monkeypatch.setattr(v_mod, "ACCOUNT_COUNT_TOLERANCE", 3)
        assert not validate_account_count(valid_accounts).passed

    def test_wrong_columns_raises_value_error(self, valid_df: pl.DataFrame) -> None:
        """Passing raw transactions DataFrame instead of derive_accounts() result raises."""
        with pytest.raises(ValueError, match="account_id"):
            validate_account_count(valid_df)

    def test_extra_column_raises_value_error(self, valid_accounts: pl.DataFrame) -> None:
        df = valid_accounts.with_columns(pl.lit(1).alias("extra"))
        with pytest.raises(ValueError, match="account_id"):
            validate_account_count(df)

    def test_empty_column_name_raises_value_error(self) -> None:
        df = pl.DataFrame({"not_account_id": ["A", "B"]})
        with pytest.raises(ValueError, match="account_id"):
            validate_account_count(df)


# ── validation: validate_laundering_ratio ────────────────────────────────────

class TestValidateLaunderingRatio:
    def test_passes_at_expected_ratio(self) -> None:
        n = 1000
        n_launder = int(round(n * v_mod.EXPECTED_LAUNDERING_RATIO))
        df = pl.DataFrame({"is_laundering": pl.Series([1] * n_launder + [0] * (n - n_launder), dtype=pl.Int8)})
        assert validate_laundering_ratio(df).passed

    def test_fails_when_ratio_too_high(self) -> None:
        df = pl.DataFrame({"is_laundering": pl.Series([1, 1, 1, 0], dtype=pl.Int8)})
        assert not validate_laundering_ratio(df).passed

    def test_fails_when_ratio_zero(self) -> None:
        df = pl.DataFrame({"is_laundering": pl.Series([0, 0, 0, 0], dtype=pl.Int8)})
        assert not validate_laundering_ratio(df).passed

    def test_ratio_within_upper_tolerance_passes(self) -> None:
        n = 10_000
        n_launder = 12  # 0.12% < 0.001 + 0.0002
        df = pl.DataFrame({"is_laundering": pl.Series([1] * n_launder + [0] * (n - n_launder), dtype=pl.Int8)})
        assert validate_laundering_ratio(df).passed

    def test_missing_column_returns_failed_result(self) -> None:
        df = pl.DataFrame({"other": [0, 0, 0]})
        result = validate_laundering_ratio(df)
        assert not result.passed
        assert result.actual == "column absent"
        assert "missing" in result.message

    def test_actual_field_is_string(self) -> None:
        df = pl.DataFrame({"is_laundering": pl.Series([0, 0, 0, 0], dtype=pl.Int8)})
        result = validate_laundering_ratio(df)
        assert "0." in str(result.actual)


# ── validation: validate_is_laundering_values ────────────────────────────────

class TestValidateIsLaunderingValues:
    def test_passes_with_only_0_and_1(self, valid_df: pl.DataFrame) -> None:
        assert validate_is_laundering_values(valid_df).passed

    def test_fails_with_unexpected_value(self, valid_df: pl.DataFrame) -> None:
        df = valid_df.with_columns(pl.lit(2).cast(pl.Int8).alias("is_laundering"))
        result = validate_is_laundering_values(df)
        assert not result.passed
        assert 2 in result.actual

    def test_passes_with_all_zeros(self) -> None:
        df = pl.DataFrame({"is_laundering": pl.Series([0, 0, 0], dtype=pl.Int8)})
        assert validate_is_laundering_values(df).passed

    def test_passes_with_all_ones(self) -> None:
        df = pl.DataFrame({"is_laundering": pl.Series([1, 1, 1], dtype=pl.Int8)})
        assert validate_is_laundering_values(df).passed

    def test_fails_when_column_absent(self) -> None:
        df = pl.DataFrame({"other": [1, 2, 3]})
        assert not validate_is_laundering_values(df).passed

    def test_uses_single_unique_pass(self, valid_df: pl.DataFrame) -> None:
        """Verify the function handles mixed valid values without error (logic test)."""
        df = valid_df.with_columns(pl.col("is_laundering").cast(pl.Int8))
        result = validate_is_laundering_values(df)
        assert result.passed
        assert result.actual == {0, 1}


# ── validation: run_all_validations ──────────────────────────────────────────

class TestRunAllValidations:
    def test_passes_and_returns_results(
        self,
        monkeypatch: pytest.MonkeyPatch,
        valid_df: pl.DataFrame,
        valid_accounts: pl.DataFrame,
    ) -> None:
        monkeypatch.setattr(v_mod, "EXPECTED_ROW_COUNT", 3)
        monkeypatch.setattr(v_mod, "EXPECTED_ACCOUNT_COUNT", 3)
        monkeypatch.setattr(v_mod, "ACCOUNT_COUNT_TOLERANCE", 0)
        monkeypatch.setattr(v_mod, "EXPECTED_LAUNDERING_RATIO", 1 / 3)
        monkeypatch.setattr(v_mod, "LAUNDERING_RATIO_TOLERANCE", 0.01)
        results = run_all_validations(valid_df, valid_accounts)
        assert all(r.passed for r in results)

    def test_raises_assertion_error_on_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        valid_df: pl.DataFrame,
        valid_accounts: pl.DataFrame,
    ) -> None:
        monkeypatch.setattr(v_mod, "EXPECTED_ROW_COUNT", 999)
        with pytest.raises(AssertionError, match="DoD gate failed"):
            run_all_validations(valid_df, valid_accounts)

    def test_error_message_names_failing_checks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        valid_df: pl.DataFrame,
        valid_accounts: pl.DataFrame,
    ) -> None:
        monkeypatch.setattr(v_mod, "EXPECTED_ROW_COUNT", 999)
        with pytest.raises(AssertionError) as exc_info:
            run_all_validations(valid_df, valid_accounts)
        assert "row_count" in str(exc_info.value)

    def test_returns_all_seven_results(
        self,
        monkeypatch: pytest.MonkeyPatch,
        valid_df: pl.DataFrame,
        valid_accounts: pl.DataFrame,
    ) -> None:
        monkeypatch.setattr(v_mod, "EXPECTED_ROW_COUNT", 3)
        monkeypatch.setattr(v_mod, "EXPECTED_ACCOUNT_COUNT", 3)
        monkeypatch.setattr(v_mod, "ACCOUNT_COUNT_TOLERANCE", 0)
        monkeypatch.setattr(v_mod, "EXPECTED_LAUNDERING_RATIO", 1 / 3)
        monkeypatch.setattr(v_mod, "LAUNDERING_RATIO_TOLERANCE", 0.01)
        results = run_all_validations(valid_df, valid_accounts)
        assert len(results) == 7

    def test_schema_failure_reported_alongside_other_failures(
        self,
        monkeypatch: pytest.MonkeyPatch,
        valid_accounts: pl.DataFrame,
    ) -> None:
        """Schema failure plus other failures all appear in the same AssertionError."""
        df_bad = pl.DataFrame({"wrong_col": [1, 2, 3]})
        monkeypatch.setattr(v_mod, "EXPECTED_ROW_COUNT", 3)
        with pytest.raises(AssertionError) as exc_info:
            run_all_validations(df_bad, valid_accounts)
        assert "schema" in str(exc_info.value)
