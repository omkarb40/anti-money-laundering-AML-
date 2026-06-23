"""
Step 4: Deterministic AML transaction rule engine.

Five typology rules operate on an account-level transaction window derived
from the HI-Small transaction DataFrame.  No pandas.  is_laundering is never
included in the window or used as a feature.  Errors propagate — there is no
silent exception swallowing inside evaluate_rules.

Public API
----------
build_account_window(df, account_id)         → pl.DataFrame
evaluate_rules(account_id, window, entity)   → list[RuleFiring]
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Optional

import polars as pl

from aml_copilot.schemas import EntityChain, RuleFiring
from aml_copilot.step4_rules.thresholds import (
    FAN_N,
    FAN_WINDOW_HOURS,
    HIGH_RISK_COUNTRIES,
    MAX_WINDOW_ROWS,
    PASSTHROUGH_MIN_RATIO,
    PASSTHROUGH_WINDOW_HOURS,
    STRUCT_BAND_HIGH,
    STRUCT_BAND_LOW,
    STRUCT_MIN_COUNT,
    STRUCT_THRESHOLD,
    STRUCT_WINDOW_HOURS,
)

# Columns the rule engine reads.  is_laundering is explicitly absent.
_WINDOW_COLS: list[str] = [
    "timestamp",
    "from_account",
    "to_account",
    "amount_paid",
    "amount_received",
    "payment_format",
]

# Subset of HIGH_RISK_COUNTRIES on the FATF blacklist (June 2023)
_FATF_BLACKLIST: frozenset[str] = frozenset({"MM", "IR", "KP"})

# Sentinel timestamp used when the account window is empty
_EPOCH = datetime(1970, 1, 1, 0, 0, 0)


# ── Window builder ────────────────────────────────────────────────────────────

def build_account_window(df: pl.DataFrame, account_id: str) -> pl.DataFrame:
    """
    Return all transactions where account_id appears as sender or receiver.

    Columns: exactly _WINDOW_COLS (is_laundering excluded regardless of input).
    Rows: sorted ascending by timestamp.
    Size: trimmed to the most recent MAX_WINDOW_ROWS rows for hub accounts.
    """
    available = [c for c in _WINDOW_COLS if c in df.columns]
    window = (
        df.filter(
            (pl.col("from_account") == account_id)
            | (pl.col("to_account") == account_id)
        )
        .select(available)
        .sort("timestamp")
    )
    if len(window) > MAX_WINDOW_ROWS:
        window = window.tail(MAX_WINDOW_ROWS)
    return window


# ── STRUCT_001 — Structuring ──────────────────────────────────────────────────

def _evaluate_structuring(
    account_id: str, window: pl.DataFrame
) -> Optional[RuleFiring]:
    lower = STRUCT_THRESHOLD * STRUCT_BAND_LOW   # inclusive lower bound
    upper = STRUCT_THRESHOLD * STRUCT_BAND_HIGH  # exclusive upper bound

    in_band = (
        window.filter(
            (pl.col("from_account") == account_id)
            & (pl.col("amount_paid") >= lower)
            & (pl.col("amount_paid") < upper)
        )
        .sort("timestamp")
    )
    if len(in_band) < STRUCT_MIN_COUNT:
        return None

    records = in_band.to_dicts()
    left = 0
    for right in range(len(records)):
        ts_right: datetime = records[right]["timestamp"]
        # Advance left pointer until window span <= STRUCT_WINDOW_HOURS
        while (
            (ts_right - records[left]["timestamp"]).total_seconds() / 3600
            > STRUCT_WINDOW_HOURS
        ):
            left += 1
        if right - left + 1 >= STRUCT_MIN_COUNT:
            sub = records[left : right + 1]
            return RuleFiring(
                rule_id="STRUCT_001",
                severity=3,
                account_id=account_id,
                evidence={
                    "txn_count": right - left + 1,
                    "amounts": [r["amount_paid"] for r in sub],
                    "threshold": STRUCT_THRESHOLD,
                    "lower_bound": lower,
                    "upper_bound": upper,
                },
                window_start=records[left]["timestamp"],
                window_end=ts_right,
            )
    return None


# ── PASSTHROUGH_001 — Rapid In-Out ────────────────────────────────────────────

def _evaluate_passthrough(
    account_id: str, window: pl.DataFrame
) -> Optional[RuleFiring]:
    inbound = window.filter(pl.col("to_account") == account_id).sort("timestamp")
    outbound = window.filter(pl.col("from_account") == account_id).sort("timestamp")

    if len(inbound) == 0 or len(outbound) == 0:
        return None

    inbound_records = inbound.to_dicts()
    outbound_records = outbound.to_dicts()
    window_secs = PASSTHROUGH_WINDOW_HOURS * 3600  # exclusive: < window_secs fires

    for inb in inbound_records:
        inb_ts: datetime = inb["timestamp"]
        inb_amount: float = inb["amount_received"]
        if inb_amount <= 0:
            continue

        # Outbound must arrive strictly after inbound and strictly within the window
        qualifying = [
            out for out in outbound_records
            if 0 < (out["timestamp"] - inb_ts).total_seconds() < window_secs
        ]
        if not qualifying:
            continue

        total_out = sum(r["amount_paid"] for r in qualifying)
        ratio = total_out / inb_amount
        if ratio >= PASSTHROUGH_MIN_RATIO:
            return RuleFiring(
                rule_id="PASSTHROUGH_001",
                severity=3,
                account_id=account_id,
                evidence={
                    "inbound_amount": inb_amount,
                    "outbound_total": round(total_out, 4),
                    "ratio": round(ratio, 4),
                    "outbound_txn_count": len(qualifying),
                    "window_hours": PASSTHROUGH_WINDOW_HOURS,
                },
                window_start=inb_ts,
                window_end=qualifying[-1]["timestamp"],
            )
    return None


# ── FAN_OUT_001 — Fan-Out ─────────────────────────────────────────────────────

def _evaluate_fan_out(
    account_id: str, window: pl.DataFrame
) -> Optional[RuleFiring]:
    outbound = window.filter(pl.col("from_account") == account_id).sort("timestamp")
    if len(outbound) < FAN_N:
        return None

    records = outbound.to_dicts()
    counter: Counter[str] = Counter()
    left = 0

    for right, row in enumerate(records):
        ts_right: datetime = row["timestamp"]
        counter[row["to_account"]] += 1

        while (
            (ts_right - records[left]["timestamp"]).total_seconds() / 3600
            > FAN_WINDOW_HOURS
        ):
            old = records[left]["to_account"]
            counter[old] -= 1
            if counter[old] == 0:
                del counter[old]
            left += 1

        if len(counter) >= FAN_N:
            sub = records[left : right + 1]
            return RuleFiring(
                rule_id="FAN_OUT_001",
                severity=2,
                account_id=account_id,
                evidence={
                    "unique_recipient_count": len(counter),
                    "recipient_ids": sorted({r["to_account"] for r in sub})[:20],
                    "txn_count": right - left + 1,
                    "window_hours": FAN_WINDOW_HOURS,
                },
                window_start=records[left]["timestamp"],
                window_end=ts_right,
            )
    return None


# ── FAN_IN_001 — Fan-In ───────────────────────────────────────────────────────

def _evaluate_fan_in(
    account_id: str, window: pl.DataFrame
) -> Optional[RuleFiring]:
    inbound = window.filter(pl.col("to_account") == account_id).sort("timestamp")
    if len(inbound) < FAN_N:
        return None

    records = inbound.to_dicts()
    counter: Counter[str] = Counter()
    left = 0

    for right, row in enumerate(records):
        ts_right: datetime = row["timestamp"]
        counter[row["from_account"]] += 1

        while (
            (ts_right - records[left]["timestamp"]).total_seconds() / 3600
            > FAN_WINDOW_HOURS
        ):
            old = records[left]["from_account"]
            counter[old] -= 1
            if counter[old] == 0:
                del counter[old]
            left += 1

        if len(counter) >= FAN_N:
            sub = records[left : right + 1]
            return RuleFiring(
                rule_id="FAN_IN_001",
                severity=2,
                account_id=account_id,
                evidence={
                    "unique_sender_count": len(counter),
                    "sender_ids": sorted({r["from_account"] for r in sub})[:20],
                    "txn_count": right - left + 1,
                    "window_hours": FAN_WINDOW_HOURS,
                },
                window_start=records[left]["timestamp"],
                window_end=ts_right,
            )
    return None


# ── CORRIDOR_001 — High-Risk Corridor ─────────────────────────────────────────

def _evaluate_corridor(
    account_id: str,
    window: pl.DataFrame,
    entity: Optional[EntityChain],
) -> Optional[RuleFiring]:
    if entity is None or entity.country is None:
        return None
    if entity.country not in HIGH_RISK_COUNTRIES:
        return None

    matched_list = (
        "FATF_BLACKLIST" if entity.country in _FATF_BLACKLIST else "HIGH_RISK"
    )
    ws: datetime = window["timestamp"].min() if len(window) > 0 else _EPOCH
    we: datetime = window["timestamp"].max() if len(window) > 0 else _EPOCH

    return RuleFiring(
        rule_id="CORRIDOR_001",
        severity=1,
        account_id=account_id,
        evidence={
            "account_country": entity.country,
            "matched_list": matched_list,
        },
        window_start=ws,
        window_end=we,
    )


# ── evaluate_rules ────────────────────────────────────────────────────────────

def evaluate_rules(
    account_id: str,
    window: pl.DataFrame,
    entity: Optional[EntityChain] = None,
) -> list[RuleFiring]:
    """
    Run all five rule evaluators against the account window.

    Errors propagate — no silent exception swallowing.

    Parameters
    ----------
    account_id : str
    window : pl.DataFrame
        Output of build_account_window(); sorted by timestamp ascending.
    entity : Optional[EntityChain]
        Required for CORRIDOR_001.  If None the corridor check is skipped.

    Returns
    -------
    list[RuleFiring]
        All firings in definition order: STRUCT, PASSTHROUGH, FAN_OUT,
        FAN_IN, CORRIDOR.  Empty list if nothing fires.
    """
    results: list[RuleFiring] = []

    f = _evaluate_structuring(account_id, window)
    if f:
        results.append(f)

    f = _evaluate_passthrough(account_id, window)
    if f:
        results.append(f)

    f = _evaluate_fan_out(account_id, window)
    if f:
        results.append(f)

    f = _evaluate_fan_in(account_id, window)
    if f:
        results.append(f)

    f = _evaluate_corridor(account_id, window, entity)
    if f:
        results.append(f)

    return results
