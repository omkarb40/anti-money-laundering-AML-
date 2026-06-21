from __future__ import annotations

import polars as pl

FAKER_SEED: int = 42


def assign_names(accounts: pl.DataFrame) -> pl.DataFrame:
    """
    Add name, country, kyc_risk columns to accounts using seeded Faker.
    Returns identity_overlay DataFrame; does not write to disk.
    """
    ...


def save_overlay(overlay: pl.DataFrame, path: str) -> None:
    """Write overlay to Parquet at path."""
    ...
