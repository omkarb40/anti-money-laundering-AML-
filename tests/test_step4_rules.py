"""
Unit and integration tests for Step 4: transaction rule engine.

Unit tests use synthetic Polars DataFrames (no disk I/O).
Integration tests require data/raw/HI-Small_Trans.csv and are auto-skipped
when the file is absent.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from aml_copilot.schemas import EntityChain, RuleFiring
from aml_copilot.step4_rules.engine import (
    build_account_window,
    evaluate_rules,
)
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

# ── Helpers ───────────────────────────────────────────────────────────────────

_BASE = datetime(2022, 1, 1, 0, 0, 0)

_LOWER = STRUCT_THRESHOLD * STRUCT_BAND_LOW   # 8 000.0
_UPPER = STRUCT_THRESHOLD * STRUCT_BAND_HIGH  # 9 900.0  (exclusive)
_IN_BAND = (_LOWER + _UPPER) / 2             # 8 950.0  — always inside [_LOWER, _UPPER)


def _ts(hour: float) -> datetime:
    return _BASE + timedelta(hours=hour)


def _make_window(rows: list[dict]) -> pl.DataFrame:
    """Build a synthetic window DataFrame from a list of row dicts.

    Required keys: from_account, to_account, hour (float offset in hours).
    Optional: amount_paid (default 0.0), amount_received (default 0.0),
              payment_format (default "Wire").
    """
    return pl.DataFrame({
        "timestamp": [_ts(r["hour"]) for r in rows],
        "from_account": [r["from_account"] for r in rows],
        "to_account": [r["to_account"] for r in rows],
        "amount_paid": [float(r.get("amount_paid", 0.0)) for r in rows],
        "amount_received": [float(r.get("amount_received", 0.0)) for r in rows],
        "payment_format": [r.get("payment_format", "Wire") for r in rows],
    }).sort("timestamp")


def _make_entity(country: str = "US", kyc_risk: str = "low") -> EntityChain:
    return EntityChain(
        account_id="ACC",
        name="Test User",
        country=country,
        kyc_risk=kyc_risk,
        hop1_counterparties=[],
        hop2_counterparties=[],
        pattern_label=None,
    )


def _rule_ids(firings: list[RuleFiring]) -> set[str]:
    return {f.rule_id for f in firings}


# ── TestBuildAccountWindow ────────────────────────────────────────────────────

class TestBuildAccountWindow:
    def _full_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "timestamp": [_ts(0), _ts(1), _ts(2)],
            "from_account": ["ACC001", "OTHER", "ACC001"],
            "to_account": ["OTHER", "ACC001", "ACC002"],
            "amount_paid": [100.0, 200.0, 300.0],
            "amount_received": [100.0, 200.0, 300.0],
            "payment_format": ["Wire", "ACH", "Wire"],
            "is_laundering": pl.Series([0, 0, 0], dtype=pl.Int8),
        })

    def test_includes_outbound_rows(self) -> None:
        w = build_account_window(self._full_df(), "ACC001")
        assert any(r["from_account"] == "ACC001" for r in w.to_dicts())

    def test_includes_inbound_rows(self) -> None:
        w = build_account_window(self._full_df(), "ACC001")
        assert any(r["to_account"] == "ACC001" for r in w.to_dicts())

    def test_excludes_unrelated_rows(self) -> None:
        w = build_account_window(self._full_df(), "ACC001")
        for r in w.to_dicts():
            assert r["from_account"] == "ACC001" or r["to_account"] == "ACC001"

    def test_sorted_by_timestamp(self) -> None:
        w = build_account_window(self._full_df(), "ACC001")
        ts = w["timestamp"].to_list()
        assert ts == sorted(ts)

    def test_is_laundering_excluded(self) -> None:
        w = build_account_window(self._full_df(), "ACC001")
        assert "is_laundering" not in w.columns

    def test_empty_result_has_correct_columns(self) -> None:
        w = build_account_window(self._full_df(), "PHANTOM")
        assert len(w) == 0
        assert "timestamp" in w.columns
        assert "from_account" in w.columns
        assert "is_laundering" not in w.columns

    def test_hub_trimmed_to_max_window_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import aml_copilot.step4_rules.engine as eng
        monkeypatch.setattr(eng, "MAX_WINDOW_ROWS", 3)
        rows = [{"from_account": "HUB", "to_account": f"R{i}", "hour": float(i)}
                for i in range(10)]
        df = pl.DataFrame({
            "timestamp": [_ts(r["hour"]) for r in rows],
            "from_account": [r["from_account"] for r in rows],
            "to_account": [r["to_account"] for r in rows],
            "amount_paid": [0.0] * 10,
            "amount_received": [0.0] * 10,
            "payment_format": ["Wire"] * 10,
        })
        w = build_account_window(df, "HUB")
        assert len(w) == 3


# ── TestStructuringRule ───────────────────────────────────────────────────────

class TestStructuringRule:
    def test_structuring_fires(self) -> None:
        """≥ STRUCT_MIN_COUNT outbound in-band txns in 24 h → STRUCT_001 at severity 3."""
        rows = [
            {"from_account": "ACC", "to_account": f"R{i}",
             "amount_paid": _IN_BAND, "hour": float(i)}
            for i in range(STRUCT_MIN_COUNT)
        ]
        firings = evaluate_rules("ACC", _make_window(rows))
        struct = [f for f in firings if f.rule_id == "STRUCT_001"]
        assert len(struct) == 1
        assert struct[0].severity == 3

    def test_structuring_no_false_fire(self) -> None:
        """Only STRUCT_MIN_COUNT - 1 qualifying txns → rule does not fire."""
        rows = [
            {"from_account": "ACC", "to_account": f"R{i}",
             "amount_paid": _IN_BAND, "hour": float(i)}
            for i in range(STRUCT_MIN_COUNT - 1)
        ]
        firings = evaluate_rules("ACC", _make_window(rows))
        assert "STRUCT_001" not in _rule_ids(firings)

    def test_no_fire_below_lower_band(self) -> None:
        """Amounts below STRUCT_BAND_LOW * STRUCT_THRESHOLD → not in band."""
        below = _LOWER - 1.0
        rows = [
            {"from_account": "ACC", "to_account": f"R{i}",
             "amount_paid": below, "hour": float(i)}
            for i in range(STRUCT_MIN_COUNT)
        ]
        firings = evaluate_rules("ACC", _make_window(rows))
        assert "STRUCT_001" not in _rule_ids(firings)

    def test_upper_band_is_exclusive(self) -> None:
        """Amount exactly at STRUCT_BAND_HIGH * STRUCT_THRESHOLD does NOT qualify."""
        rows = [
            {"from_account": "ACC", "to_account": f"R{i}",
             "amount_paid": _UPPER, "hour": float(i)}  # exactly 9900.0
            for i in range(STRUCT_MIN_COUNT)
        ]
        firings = evaluate_rules("ACC", _make_window(rows))
        assert "STRUCT_001" not in _rule_ids(firings)

    def test_no_fire_inbound_only(self) -> None:
        """Inbound transactions in the band do not trigger structuring."""
        rows = [
            {"from_account": f"S{i}", "to_account": "ACC",
             "amount_paid": _IN_BAND, "amount_received": _IN_BAND, "hour": float(i)}
            for i in range(STRUCT_MIN_COUNT)
        ]
        firings = evaluate_rules("ACC", _make_window(rows))
        assert "STRUCT_001" not in _rule_ids(firings)

    def test_no_fire_outside_24h_window(self) -> None:
        """Qualifying txns spread beyond STRUCT_WINDOW_HOURS → do not form a valid window."""
        # Place txns at 0h, 13h, 26h — no three fall within 24h span
        rows = [
            {"from_account": "ACC", "to_account": f"R{i}",
             "amount_paid": _IN_BAND, "hour": float(i * 13)}
            for i in range(STRUCT_MIN_COUNT)
        ]
        firings = evaluate_rules("ACC", _make_window(rows))
        assert "STRUCT_001" not in _rule_ids(firings)

    def test_evidence_keys(self) -> None:
        rows = [
            {"from_account": "ACC", "to_account": f"R{i}",
             "amount_paid": _IN_BAND, "hour": float(i)}
            for i in range(STRUCT_MIN_COUNT)
        ]
        firings = evaluate_rules("ACC", _make_window(rows))
        struct = next(f for f in firings if f.rule_id == "STRUCT_001")
        for key in ("txn_count", "amounts", "threshold", "lower_bound", "upper_bound"):
            assert key in struct.evidence

    def test_account_id_in_firing(self) -> None:
        rows = [
            {"from_account": "MYACC", "to_account": f"R{i}",
             "amount_paid": _IN_BAND, "hour": float(i)}
            for i in range(STRUCT_MIN_COUNT)
        ]
        firings = evaluate_rules("MYACC", _make_window(rows))
        struct = next(f for f in firings if f.rule_id == "STRUCT_001")
        assert struct.account_id == "MYACC"


# ── TestPassthroughRule ───────────────────────────────────────────────────────

class TestPassthroughRule:
    def test_passthrough_fires(self) -> None:
        """≥ 80% of inbound forwarded within 24 h → PASSTHROUGH_001 at severity 3."""
        rows = [
            {"from_account": "SRC", "to_account": "ACC",
             "amount_paid": 0.0, "amount_received": 10_000.0, "hour": 0.0},
            {"from_account": "ACC", "to_account": "DST",
             "amount_paid": 9_000.0, "amount_received": 0.0, "hour": 1.0},
        ]
        firings = evaluate_rules("ACC", _make_window(rows))
        pt = [f for f in firings if f.rule_id == "PASSTHROUGH_001"]
        assert len(pt) == 1
        assert pt[0].severity == 3

    def test_passthrough_time_boundary(self) -> None:
        """Outbound at exactly PASSTHROUGH_WINDOW_HOURS does NOT fire; at -1 s does."""
        inb_amount = 10_000.0
        out_amount = 9_000.0  # 90% — above threshold

        # Case 1: outbound at exactly 24 h → must NOT fire
        rows_exact = [
            {"from_account": "SRC", "to_account": "ACC",
             "amount_received": inb_amount, "amount_paid": 0.0, "hour": 0.0},
            {"from_account": "ACC", "to_account": "DST",
             "amount_paid": out_amount, "amount_received": 0.0,
             "hour": PASSTHROUGH_WINDOW_HOURS},  # exactly 24 h
        ]
        firings_exact = evaluate_rules("ACC", _make_window(rows_exact))
        assert "PASSTHROUGH_001" not in _rule_ids(firings_exact), (
            "PASSTHROUGH_001 fired at exactly PASSTHROUGH_WINDOW_HOURS — "
            "boundary must be exclusive"
        )

        # Case 2: outbound 1 second before the boundary → MUST fire
        rows_just_inside = [
            {"from_account": "SRC", "to_account": "ACC",
             "amount_received": inb_amount, "amount_paid": 0.0, "hour": 0.0},
        ]
        # Build manually to get exact 23:59 offset
        inb_ts = _ts(0.0)
        almost_24h = inb_ts + timedelta(hours=PASSTHROUGH_WINDOW_HOURS) - timedelta(seconds=1)
        df = pl.DataFrame({
            "timestamp": [inb_ts, almost_24h],
            "from_account": ["SRC", "ACC"],
            "to_account": ["ACC", "DST"],
            "amount_paid": [0.0, out_amount],
            "amount_received": [inb_amount, 0.0],
            "payment_format": ["Wire", "Wire"],
        }).sort("timestamp")
        firings_inside = evaluate_rules("ACC", df)
        assert "PASSTHROUGH_001" in _rule_ids(firings_inside), (
            "PASSTHROUGH_001 did not fire with outbound 1 s inside window"
        )

    def test_no_fire_below_ratio(self) -> None:
        """Outbound below PASSTHROUGH_MIN_RATIO → rule does not fire."""
        below_ratio = PASSTHROUGH_MIN_RATIO - 0.01
        rows = [
            {"from_account": "SRC", "to_account": "ACC",
             "amount_received": 10_000.0, "hour": 0.0},
            {"from_account": "ACC", "to_account": "DST",
             "amount_paid": 10_000.0 * below_ratio, "hour": 1.0},
        ]
        firings = evaluate_rules("ACC", _make_window(rows))
        assert "PASSTHROUGH_001" not in _rule_ids(firings)

    def test_no_fire_outbound_before_inbound(self) -> None:
        """Outbound that precedes inbound is not counted."""
        rows = [
            {"from_account": "ACC", "to_account": "DST",
             "amount_paid": 9_000.0, "hour": 0.0},       # outbound first
            {"from_account": "SRC", "to_account": "ACC",
             "amount_received": 10_000.0, "hour": 1.0},  # inbound after
        ]
        firings = evaluate_rules("ACC", _make_window(rows))
        assert "PASSTHROUGH_001" not in _rule_ids(firings)

    def test_no_fire_no_outbound(self) -> None:
        """Inbound with no subsequent outbound → rule does not fire."""
        rows = [
            {"from_account": "SRC", "to_account": "ACC",
             "amount_received": 10_000.0, "hour": 0.0},
        ]
        firings = evaluate_rules("ACC", _make_window(rows))
        assert "PASSTHROUGH_001" not in _rule_ids(firings)

    def test_evidence_ratio_field(self) -> None:
        rows = [
            {"from_account": "SRC", "to_account": "ACC",
             "amount_received": 10_000.0, "hour": 0.0},
            {"from_account": "ACC", "to_account": "DST",
             "amount_paid": 9_000.0, "hour": 1.0},
        ]
        firings = evaluate_rules("ACC", _make_window(rows))
        pt = next(f for f in firings if f.rule_id == "PASSTHROUGH_001")
        assert "ratio" in pt.evidence
        assert pt.evidence["ratio"] == pytest.approx(0.9, rel=1e-3)


# ── TestFanOutRule ────────────────────────────────────────────────────────────

class TestFanOutRule:
    def test_fan_out_fires(self) -> None:
        """≥ FAN_N unique recipients within FAN_WINDOW_HOURS → FAN_OUT_001 at severity 2."""
        rows = [
            {"from_account": "HUB", "to_account": f"R{i}", "hour": float(i)}
            for i in range(FAN_N + 1)  # one above FAN_N
        ]
        firings = evaluate_rules("HUB", _make_window(rows))
        fan = [f for f in firings if f.rule_id == "FAN_OUT_001"]
        assert len(fan) == 1
        assert fan[0].severity == 2

    def test_no_fire_fan_n_minus_1(self) -> None:
        rows = [
            {"from_account": "HUB", "to_account": f"R{i}", "hour": float(i)}
            for i in range(FAN_N - 1)
        ]
        firings = evaluate_rules("HUB", _make_window(rows))
        assert "FAN_OUT_001" not in _rule_ids(firings)

    def test_deduplicates_recipients(self) -> None:
        """Sending to the same recipient FAN_N times is only 1 unique — does not fire."""
        rows = [
            {"from_account": "HUB", "to_account": "SAME", "hour": float(i)}
            for i in range(FAN_N + 1)
        ]
        firings = evaluate_rules("HUB", _make_window(rows))
        assert "FAN_OUT_001" not in _rule_ids(firings)

    def test_fan_n_unique_outside_window_does_not_fire(self) -> None:
        """FAN_N unique recipients spread over 2 × FAN_WINDOW_HOURS → no window contains enough."""
        step = FAN_WINDOW_HOURS / (FAN_N - 1) + 0.5  # space each txn beyond equal share
        rows = [
            {"from_account": "HUB", "to_account": f"R{i}", "hour": i * step}
            for i in range(FAN_N)
        ]
        # First and last are > FAN_WINDOW_HOURS apart, so no window of FAN_WINDOW_HOURS
        # can contain all FAN_N unique recipients.
        firings = evaluate_rules("HUB", _make_window(rows))
        assert "FAN_OUT_001" not in _rule_ids(firings)

    def test_evidence_recipient_count(self) -> None:
        rows = [
            {"from_account": "HUB", "to_account": f"R{i}", "hour": float(i)}
            for i in range(FAN_N + 2)
        ]
        firings = evaluate_rules("HUB", _make_window(rows))
        fan = next(f for f in firings if f.rule_id == "FAN_OUT_001")
        assert fan.evidence["unique_recipient_count"] >= FAN_N
        assert "recipient_ids" in fan.evidence
        assert len(fan.evidence["recipient_ids"]) <= 20

    def test_account_id_in_firing(self) -> None:
        rows = [
            {"from_account": "MYACC", "to_account": f"R{i}", "hour": float(i)}
            for i in range(FAN_N + 1)
        ]
        firings = evaluate_rules("MYACC", _make_window(rows))
        fan = next(f for f in firings if f.rule_id == "FAN_OUT_001")
        assert fan.account_id == "MYACC"


# ── TestFanInRule ─────────────────────────────────────────────────────────────

class TestFanInRule:
    def test_fan_in_fires(self) -> None:
        """≥ FAN_N unique senders within FAN_WINDOW_HOURS → FAN_IN_001 at severity 2."""
        rows = [
            {"from_account": f"S{i}", "to_account": "HUB",
             "amount_received": 1_000.0, "hour": float(i)}
            for i in range(FAN_N + 1)
        ]
        firings = evaluate_rules("HUB", _make_window(rows))
        fan = [f for f in firings if f.rule_id == "FAN_IN_001"]
        assert len(fan) == 1
        assert fan[0].severity == 2

    def test_no_fire_fan_n_minus_1_senders(self) -> None:
        rows = [
            {"from_account": f"S{i}", "to_account": "HUB",
             "amount_received": 1_000.0, "hour": float(i)}
            for i in range(FAN_N - 1)
        ]
        firings = evaluate_rules("HUB", _make_window(rows))
        assert "FAN_IN_001" not in _rule_ids(firings)

    def test_deduplicates_senders(self) -> None:
        """Same sender sending FAN_N times is only 1 unique — does not fire."""
        rows = [
            {"from_account": "SAME", "to_account": "HUB",
             "amount_received": 1_000.0, "hour": float(i)}
            for i in range(FAN_N + 1)
        ]
        firings = evaluate_rules("HUB", _make_window(rows))
        assert "FAN_IN_001" not in _rule_ids(firings)

    def test_evidence_sender_count(self) -> None:
        rows = [
            {"from_account": f"S{i}", "to_account": "HUB",
             "amount_received": 1_000.0, "hour": float(i)}
            for i in range(FAN_N + 2)
        ]
        firings = evaluate_rules("HUB", _make_window(rows))
        fan = next(f for f in firings if f.rule_id == "FAN_IN_001")
        assert fan.evidence["unique_sender_count"] >= FAN_N
        assert "sender_ids" in fan.evidence
        assert len(fan.evidence["sender_ids"]) <= 20


# ── TestCorridorRule ──────────────────────────────────────────────────────────

class TestCorridorRule:
    def test_corridor_fires_high_risk_country(self) -> None:
        """Entity in HIGH_RISK_COUNTRIES → CORRIDOR_001 fires at severity 1."""
        country = next(iter(HIGH_RISK_COUNTRIES))  # pick any listed country
        entity = _make_entity(country=country)
        window = _make_window([
            {"from_account": "ACC", "to_account": "X", "hour": 0.0},
        ])
        firings = evaluate_rules("ACC", window, entity)
        corridor = [f for f in firings if f.rule_id == "CORRIDOR_001"]
        assert len(corridor) == 1
        assert corridor[0].severity == 1

    def test_no_fire_safe_country(self) -> None:
        entity = _make_entity(country="US")
        window = _make_window([
            {"from_account": "ACC", "to_account": "X", "hour": 0.0},
        ])
        firings = evaluate_rules("ACC", window, entity)
        assert "CORRIDOR_001" not in _rule_ids(firings)

    def test_no_fire_none_entity(self) -> None:
        window = _make_window([
            {"from_account": "ACC", "to_account": "X", "hour": 0.0},
        ])
        firings = evaluate_rules("ACC", window, entity=None)
        assert "CORRIDOR_001" not in _rule_ids(firings)

    def test_corridor_ir_is_fatf_blacklist(self) -> None:
        """Iran (IR) should be classified as FATF_BLACKLIST in evidence."""
        entity = _make_entity(country="IR")
        window = _make_window([
            {"from_account": "ACC", "to_account": "X", "hour": 0.0},
        ])
        firings = evaluate_rules("ACC", window, entity)
        corridor = next(f for f in firings if f.rule_id == "CORRIDOR_001")
        assert corridor.evidence["matched_list"] == "FATF_BLACKLIST"

    def test_evidence_has_account_country(self) -> None:
        entity = _make_entity(country="IR")
        window = _make_window([
            {"from_account": "ACC", "to_account": "X", "hour": 0.0},
        ])
        firings = evaluate_rules("ACC", window, entity)
        corridor = next(f for f in firings if f.rule_id == "CORRIDOR_001")
        assert corridor.evidence["account_country"] == "IR"

    def test_corridor_fires_on_empty_window(self) -> None:
        """CORRIDOR_001 fires even with no transactions — country risk is static."""
        entity = _make_entity(country="IR")
        empty = pl.DataFrame({
            "timestamp": pl.Series([], dtype=pl.Datetime("us")),
            "from_account": pl.Series([], dtype=pl.Utf8),
            "to_account": pl.Series([], dtype=pl.Utf8),
            "amount_paid": pl.Series([], dtype=pl.Float64),
            "amount_received": pl.Series([], dtype=pl.Float64),
            "payment_format": pl.Series([], dtype=pl.Utf8),
        })
        firings = evaluate_rules("ACC", empty, entity)
        assert "CORRIDOR_001" in _rule_ids(firings)


# ── TestEvaluateRules ─────────────────────────────────────────────────────────

class TestEvaluateRules:
    def test_returns_empty_on_clean_window(self) -> None:
        """Normal single transaction with no typology signals → empty list."""
        window = _make_window([
            {"from_account": "ACC", "to_account": "DST",
             "amount_paid": 500.0, "hour": 0.0},
        ])
        firings = evaluate_rules("ACC", window)
        assert firings == []

    def test_empty_window_returns_empty(self) -> None:
        empty = pl.DataFrame({
            "timestamp": pl.Series([], dtype=pl.Datetime("us")),
            "from_account": pl.Series([], dtype=pl.Utf8),
            "to_account": pl.Series([], dtype=pl.Utf8),
            "amount_paid": pl.Series([], dtype=pl.Float64),
            "amount_received": pl.Series([], dtype=pl.Float64),
            "payment_format": pl.Series([], dtype=pl.Utf8),
        })
        firings = evaluate_rules("ACC", empty)
        assert firings == []

    def test_all_five_rules_fire_simultaneously(self) -> None:
        """Construct a window that triggers all five rules at once."""
        # 5 inbound senders (FAN_IN) each contributing to PASSTHROUGH ratio
        # 5 outbound recipients (FAN_OUT) all in the structuring band (STRUCT)
        # Entity in HIGH_RISK_COUNTRIES (CORRIDOR)
        rows = (
            [{"from_account": f"S{i}", "to_account": "ACC",
              "amount_received": 50_000.0, "hour": float(i)}
             for i in range(FAN_N)]           # FAN_IN: FAN_N unique senders
            + [{"from_account": "ACC", "to_account": f"R{j}",
                "amount_paid": _IN_BAND, "hour": float(FAN_N + j)}
               for j in range(FAN_N + 1)]    # FAN_OUT + STRUCT (FAN_N+1 outbound)
        )
        # PASSTHROUGH: first inbound 50000, outbound total = (FAN_N+1) * _IN_BAND > 80%
        window = _make_window(rows)
        entity = _make_entity(country="IR")

        firings = evaluate_rules("ACC", window, entity)
        fired = _rule_ids(firings)

        assert "STRUCT_001" in fired, f"STRUCT_001 missing; fired: {fired}"
        assert "PASSTHROUGH_001" in fired, f"PASSTHROUGH_001 missing; fired: {fired}"
        assert "FAN_OUT_001" in fired, f"FAN_OUT_001 missing; fired: {fired}"
        assert "FAN_IN_001" in fired, f"FAN_IN_001 missing; fired: {fired}"
        assert "CORRIDOR_001" in fired, f"CORRIDOR_001 missing; fired: {fired}"

    def test_severities_correct(self) -> None:
        rows = (
            [{"from_account": f"S{i}", "to_account": "ACC",
              "amount_received": 50_000.0, "hour": float(i)}
             for i in range(FAN_N)]
            + [{"from_account": "ACC", "to_account": f"R{j}",
                "amount_paid": _IN_BAND, "hour": float(FAN_N + j)}
               for j in range(FAN_N + 1)]
        )
        firings = evaluate_rules("ACC", _make_window(rows), _make_entity("IR"))
        severity_by_rule = {f.rule_id: f.severity for f in firings}
        assert severity_by_rule.get("STRUCT_001") == 3
        assert severity_by_rule.get("PASSTHROUGH_001") == 3
        assert severity_by_rule.get("FAN_OUT_001") == 2
        assert severity_by_rule.get("FAN_IN_001") == 2
        assert severity_by_rule.get("CORRIDOR_001") == 1

    def test_account_id_in_all_firings(self) -> None:
        rows = [
            {"from_account": "MYACC", "to_account": f"R{i}", "hour": float(i)}
            for i in range(FAN_N + 1)
        ]
        firings = evaluate_rules("MYACC", _make_window(rows))
        assert all(f.account_id == "MYACC" for f in firings)

    def test_rule_firings_are_rule_firing_instances(self) -> None:
        rows = [
            {"from_account": "ACC", "to_account": f"R{i}", "hour": float(i)}
            for i in range(FAN_N + 1)
        ]
        firings = evaluate_rules("ACC", _make_window(rows))
        assert all(isinstance(f, RuleFiring) for f in firings)

    def test_window_timestamps_span_triggering_txns(self) -> None:
        """window_start <= window_end for every returned RuleFiring."""
        rows = [
            {"from_account": "ACC", "to_account": f"R{i}", "hour": float(i)}
            for i in range(FAN_N + 1)
        ]
        firings = evaluate_rules("ACC", _make_window(rows))
        for f in firings:
            assert f.window_start <= f.window_end


# ── TestThresholds ────────────────────────────────────────────────────────────

class TestThresholds:
    def test_thresholds_checksum(self) -> None:
        """SHA-256 of thresholds.py matches the committed value in checksums.sha256."""
        from aml_copilot.utils.checksum import compute_sha256

        project_root = Path(__file__).resolve().parents[1]
        thresholds_file = (
            project_root / "src" / "aml_copilot" / "step4_rules" / "thresholds.py"
        )
        checksum_file = project_root / "artifacts" / "checksums.sha256"

        assert checksum_file.exists(), (
            f"checksums.sha256 not found at {checksum_file}"
        )
        from aml_copilot.utils.checksum import _to_key
        current_digest = compute_sha256(thresholds_file)
        key = _to_key(thresholds_file)

        for line in checksum_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("  ", 1)
            if len(parts) == 2 and parts[1] == key:
                assert parts[0] == current_digest, (
                    f"thresholds.py has been modified after freeze!\n"
                    f"Committed : {parts[0]}\n"
                    f"Current   : {current_digest}"
                )
                return

        pytest.fail(
            f"No checksum entry found for thresholds.py in {checksum_file}.\n"
            f"Run: python -c \"from aml_copilot.utils.checksum import append_checksum; "
            f"append_checksum('{thresholds_file}', '{checksum_file}')\""
        )

    def test_struct_band_sanity(self) -> None:
        assert 0 < STRUCT_BAND_LOW < STRUCT_BAND_HIGH < 1.0
        assert STRUCT_THRESHOLD > 0
        assert STRUCT_MIN_COUNT >= 2

    def test_passthrough_ratio_between_0_and_1(self) -> None:
        assert 0 < PASSTHROUGH_MIN_RATIO < 1.0

    def test_fan_n_positive(self) -> None:
        assert FAN_N >= 2

    def test_high_risk_countries_nonempty_frozenset(self) -> None:
        assert isinstance(HIGH_RISK_COUNTRIES, frozenset)
        assert len(HIGH_RISK_COUNTRIES) > 0


# ── TestIntegration ───────────────────────────────────────────────────────────

@pytest.mark.integration
class TestIntegration:
    _TRANS = Path("data/raw/HI-Small_Trans.csv")
    _FAN_OUT_HUB = "800737690"   # known FAN-OUT hub from HI-Small_Patterns.txt
    _FAN_IN_HUB = "811ED7DF0"    # known FAN-IN hub with 20 unique senders in 24 h

    @pytest.fixture(autouse=True)
    def require_data(self) -> None:
        if not self._TRANS.exists():
            pytest.skip(f"Transaction CSV not found: {self._TRANS}")

    @pytest.fixture
    def real_df(self) -> pl.DataFrame:
        from aml_copilot.step0_scaffold.data_loader import load_transactions
        return load_transactions(self._TRANS)

    def test_fan_out_fires_on_known_hub(self, real_df: pl.DataFrame) -> None:
        window = build_account_window(real_df, self._FAN_OUT_HUB)
        firings = evaluate_rules(self._FAN_OUT_HUB, window)
        fan = [f for f in firings if f.rule_id == "FAN_OUT_001"]
        assert len(fan) == 1, (
            f"FAN_OUT_001 did not fire on {self._FAN_OUT_HUB}. "
            f"Rules fired: {_rule_ids(firings)}"
        )
        assert fan[0].evidence["unique_recipient_count"] >= FAN_N

    def test_fan_in_fires_on_known_hub(self, real_df: pl.DataFrame) -> None:
        window = build_account_window(real_df, self._FAN_IN_HUB)
        firings = evaluate_rules(self._FAN_IN_HUB, window)
        fan = [f for f in firings if f.rule_id == "FAN_IN_001"]
        assert len(fan) == 1, (
            f"FAN_IN_001 did not fire on {self._FAN_IN_HUB}. "
            f"Rules fired: {_rule_ids(firings)}"
        )
        assert fan[0].evidence["unique_sender_count"] >= FAN_N

    def test_isolated_account_no_crash(self, real_df: pl.DataFrame) -> None:
        """Account with no transactions produces empty firing list without raising."""
        window = build_account_window(real_df, "PHANTOM_ACCOUNT_XYZ")
        firings = evaluate_rules("PHANTOM_ACCOUNT_XYZ", window)
        assert firings == []

    def test_window_excludes_is_laundering(self, real_df: pl.DataFrame) -> None:
        window = build_account_window(real_df, self._FAN_OUT_HUB)
        assert "is_laundering" not in window.columns
