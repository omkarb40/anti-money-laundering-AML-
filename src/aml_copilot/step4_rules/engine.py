from __future__ import annotations

import polars as pl

from aml_copilot.schemas import RuleFiring


def evaluate_structuring(account_id: str, window: pl.DataFrame) -> list[RuleFiring]:
    """STRUCT_001 — severity 3. Multiple sub-threshold txns in STRUCT_WINDOW_HOURS."""
    ...


def evaluate_passthrough(account_id: str, window: pl.DataFrame) -> list[RuleFiring]:
    """PASSTHROUGH_001 — severity 3. Inbound then outbound >= PASSTHROUGH_MIN_RATIO within window."""
    ...


def evaluate_fan_out(account_id: str, window: pl.DataFrame) -> list[RuleFiring]:
    """FAN_OUT_001 — severity 2. >= FAN_N unique recipients in FAN_WINDOW_HOURS."""
    ...


def evaluate_fan_in(account_id: str, window: pl.DataFrame) -> list[RuleFiring]:
    """FAN_IN_001 — severity 2. >= FAN_N unique senders in FAN_WINDOW_HOURS."""
    ...


def evaluate_cycle(
    account_id: str,
    adjacency: dict[str, set[str]],
) -> list[RuleFiring]:
    """CYCLE_001 — severity 3. Account participates in cycle of length <= CYCLE_MAX_LEN."""
    ...


def evaluate_bipartite(account_id: str, window: pl.DataFrame) -> list[RuleFiring]:
    """BIPARTITE_001 — severity 2. Fan-out then fan-in to overlapping counterparty set."""
    ...


def evaluate_corridor(account_id: str, window: pl.DataFrame) -> list[RuleFiring]:
    """CORRIDOR_001 — severity 1. Counterparty in HIGH_RISK_COUNTRIES."""
    ...


def evaluate_velocity(account_id: str, window: pl.DataFrame) -> list[RuleFiring]:
    """VELOCITY_001 — severity 1. Transaction count > VELOCITY_N in VELOCITY_WINDOW_HOURS."""
    ...


def run_all_rules(
    account_id: str,
    window: pl.DataFrame,
    adjacency: dict[str, set[str]],
) -> list[RuleFiring]:
    """Run all 8 rule evaluators and return the combined list of firings."""
    ...
