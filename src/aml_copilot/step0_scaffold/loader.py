from __future__ import annotations

import polars as pl


def load_transactions(path: str) -> pl.DataFrame:
    """Load HI-Small_Trans.csv as a Polars DataFrame with typed columns."""
    ...


def derive_accounts(df: pl.DataFrame) -> pl.DataFrame:
    """Return deduplicated union of From/To Account IDs as a single-column DataFrame."""
    ...


def load_patterns(path: str) -> dict[str, list[str]]:
    """Parse HI-Small_Patterns.txt → {typology_name: [account_id, ...]}."""
    ...
