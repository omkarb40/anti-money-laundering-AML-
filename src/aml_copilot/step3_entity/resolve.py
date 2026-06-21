from __future__ import annotations

import polars as pl

from aml_copilot.schemas import EntityChain

HOP2_CAP: int = 50   # max hop-2 neighbours returned
MAX_DEPTH: int = 10  # cycle guard


def build_adjacency(df: pl.DataFrame) -> dict[str, set[str]]:
    """Build undirected adjacency dict {account_id: {neighbour_ids}} from transaction DataFrame."""
    ...


def resolve_entity(
    account_id: str,
    adjacency: dict[str, set[str]],
    overlay: pl.DataFrame,
    patterns: dict[str, list[str]],
) -> EntityChain:
    """
    Return EntityChain for account_id.
    hop2_counterparties capped at HOP2_CAP; cycle guard at MAX_DEPTH.
    pattern_label set if account_id appears in patterns dict.
    """
    ...
