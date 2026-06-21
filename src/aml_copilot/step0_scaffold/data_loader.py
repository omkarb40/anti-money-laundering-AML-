"""
Load IBM AMLSim HI-Small transaction data using Polars.

HI-Small_Trans.csv has a duplicate "Account" column header — the sender and receiver
account columns are both named "Account" in the raw file. Polars renames the second
occurrence to "Account_duplicated_0" with has_header=True. We bypass this by using
has_header=False, skip_rows=1, and new_columns to assign canonical names directly.

Raw header (11 columns):
    Timestamp, From Bank, Account, To Bank, Account, Amount Received,
    Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering
"""
from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl

COLUMN_NAMES: list[str] = [
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

# Ordered by likelihood for IBM AMLSim; detection tries each in turn.
TIMESTAMP_FORMATS: list[str] = [
    "%Y/%m/%d %H:%M",      # most common IBM AMLSim HI-Small format
    "%Y/%m/%d %H:%M:%S",   # with seconds
    "%Y-%m-%d %H:%M",      # ISO-style
    "%Y-%m-%d %H:%M:%S",   # ISO with seconds
]


# ── Private helpers ───────────────────────────────────────────────────────────

def _detect_timestamp_format(path: Path) -> str:
    """
    Read the first data row and probe TIMESTAMP_FORMATS in order.
    Returns the first format that successfully parses the sample value.

    Raises
    ------
    ValueError
        If the file has no data rows, or if no known format matches —
        with the sample value and tried formats in the message.
    """
    with open(path, encoding="utf-8") as fh:
        fh.readline()  # skip header row
        first_data_row = fh.readline().strip()

    if not first_data_row:
        raise ValueError(
            f"No data rows found in {path}. "
            "The file may contain only a header or be empty."
        )

    sample_ts = first_data_row.split(",")[0].strip()

    for fmt in TIMESTAMP_FORMATS:
        try:
            pl.Series([sample_ts]).str.to_datetime(format=fmt, strict=True)
            return fmt
        except Exception:
            continue

    raise ValueError(
        f"Timestamp value {sample_ts!r} in {path} did not match any known format.\n"
        f"Tried: {TIMESTAMP_FORMATS}\n"
        f"To support a new format, append it to TIMESTAMP_FORMATS in data_loader.py."
    )


# ── Public API ────────────────────────────────────────────────────────────────

def inspect_raw_csv(path: str | Path, n_rows: int = 5) -> None:
    """
    Print exact raw column names and first n_rows data rows without any processing.

    Also shows how Polars renames duplicate column headers when has_header=True,
    so the reader can understand why has_header=False + new_columns is required.

    Example
    -------
        python -c "
        from aml_copilot.step0_scaffold.data_loader import inspect_raw_csv
        inspect_raw_csv('data/raw/HI-Small_Trans.csv')
        "

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, encoding="utf-8") as fh:
        header_line = fh.readline().rstrip("\n")
        raw_cols = header_line.split(",")

        print("=== Raw column names ===")
        for i, col in enumerate(raw_cols):
            print(f"  [{i:2d}] {col!r}")

        # Simulate Polars' has_header=True duplicate renaming
        seen: dict[str, int] = {}
        renamed: list[str] = []
        for col in raw_cols:
            if col in seen:
                renamed.append(f"{col}_duplicated_{seen[col]}")
                seen[col] += 1
            else:
                renamed.append(col)
                seen[col] = 0

        if renamed != raw_cols:
            print("\n=== Polars auto-rename with has_header=True (duplicates only) ===")
            for i, (raw, new) in enumerate(zip(raw_cols, renamed)):
                if raw != new:
                    print(f"  [{i:2d}] {raw!r}  →  {new!r}")

        print(f"\n=== First {n_rows} data rows (raw) ===")
        for i, line in enumerate(fh):
            if i >= n_rows:
                break
            print(f"  row {i + 1}: {line.rstrip()}")


def load_transactions(path: str | Path) -> pl.DataFrame:
    """
    Load HI-Small_Trans.csv and return a typed Polars DataFrame.

    scan_csv with collect() performs lazy query planning and then materialises
    the full result into memory. This is NOT streaming — the entire DataFrame
    is held in RAM after collect(). For true streaming use
    .collect(streaming=True) (requires Polars >= 0.19 with the streaming engine).

    account columns (from_account, to_account) are kept as String (pl.String).
    infer_schema_length=0 forces Polars to read all columns as String before
    explicit casts, which prevents numeric account IDs from being inferred as
    Int64 and silently destroying any leading zeros.

    The timestamp format is detected from the first data row so that format
    mismatches produce a helpful error rather than a generic parse failure.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If the timestamp in the first data row does not match any known format.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Transactions file not found: {path}\n"
            "Expected HI-Small_Trans.csv from IBM AMLSim. "
            "Place it under data/raw/ or set TRANSACTIONS_PATH."
        )

    timestamp_format = _detect_timestamp_format(path)

    return (
        pl.scan_csv(
            path,
            has_header=False,
            skip_rows=1,
            new_columns=COLUMN_NAMES,
            infer_schema_length=0,      # all columns read as String; prevents Int64
            null_values=["", "NA", "null"],
        )
        .with_columns(
            pl.col("timestamp").str.to_datetime(
                format=timestamp_format, strict=True
            ),
            pl.col("amount_received").cast(pl.Float64),
            pl.col("amount_paid").cast(pl.Float64),
            pl.col("is_laundering").cast(pl.Int8),
            # from_account and to_account remain pl.String — no cast needed
        )
        .collect()
    )


def derive_accounts(df: pl.DataFrame) -> pl.DataFrame:
    """
    Return a single-column DataFrame of unique account IDs (union of from_account
    and to_account). Column name is "account_id".

    Uses Series concat (not DataFrame.select) to avoid creating two full-column
    DataFrame copies before concatenation. Peak overhead is one concatenated
    Series of 2 × len(df) elements, reduced to unique count by .unique().
    """
    return (
        pl.concat(
            [
                df["from_account"].alias("account_id"),
                df["to_account"].alias("account_id"),
            ]
        )
        .unique()
        .to_frame()
    )


def load_patterns(path: str | Path) -> dict[str, list[str]]:
    """
    Parse HI-Small_Patterns.txt into {typology: [account_id, ...]}.

    Expected format — one pattern per line, whitespace or comma separated:
        <typology_name>  <account_id_1>  <account_id_2>  ...

    Lines starting with '#' are treated as comments and skipped.
    Lines with fewer than 2 tokens are skipped with a UserWarning.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Patterns file not found: {path}")

    patterns: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.replace(",", " ").split()
            if len(tokens) < 2:
                warnings.warn(
                    f"Patterns line {lineno} has only {len(tokens)} token(s); "
                    f"skipped: {line!r}",
                    stacklevel=2,
                )
                continue
            typology, *accounts = tokens
            patterns.setdefault(typology, []).extend(accounts)

    return patterns
