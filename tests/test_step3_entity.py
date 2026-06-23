"""
Unit and integration tests for Step 3: entity resolution.

Unit tests (TestBuildTransactionGraph, TestBuildOverlayMap, TestBuildPatternMap,
TestParsePatternFile, TestResolveEntity) require no disk I/O — all use synthetic
DataFrames and in-memory dicts.

Integration tests (TestPatternChainIntegration) require:
  data/raw/HI-Small_Trans.csv
  data/raw/HI-Small_Patterns.txt
  data/processed/identity_overlay.parquet
and are auto-skipped when those files are absent.
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pytest

from aml_copilot.step3_entity.resolve import (
    HOP2_CAP,
    build_overlay_map,
    build_pattern_map,
    build_transaction_graph,
    parse_patterns_file,
    resolve_entity,
)


# ── Module-level fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def cycle_df() -> pl.DataFrame:
    """3-cycle: ACC001→ACC002→ACC003→ACC001."""
    return pl.DataFrame({
        "from_account": ["ACC001", "ACC002", "ACC003"],
        "to_account":   ["ACC002", "ACC003", "ACC001"],
    })


@pytest.fixture
def fanout_df() -> pl.DataFrame:
    """Fan-out + one hop-2 leaf. HUB→R1,R2,R3; R2→LEAF."""
    return pl.DataFrame({
        "from_account": ["HUB", "HUB", "HUB", "R2"],
        "to_account":   ["R1",  "R2",  "R3",  "LEAF"],
    })


@pytest.fixture
def hub_df() -> pl.DataFrame:
    """Hub with 75 hop-2 neighbours (above the 50-cap)."""
    from_accounts: list[str] = ["HUB"] * 5
    to_accounts: list[str] = [f"R{i:02d}" for i in range(1, 6)]
    for i in range(1, 6):
        for j in range(1, 16):  # 15 downstream per receiver = 75 hop-2 total
            from_accounts.append(f"R{i:02d}")
            to_accounts.append(f"R{i:02d}_D{j:02d}")
    return pl.DataFrame({"from_account": from_accounts, "to_account": to_accounts})


@pytest.fixture
def simple_overlay_map() -> dict[str, dict]:
    accounts = (
        ["ACC001", "ACC002", "ACC003", "ACC004", "ACC005",
         "HUB", "R1", "R2", "R3", "LEAF", "ISOLATED"]
        + [f"R{i:02d}" for i in range(1, 6)]
        + [f"R{i:02d}_D{j:02d}" for i in range(1, 6) for j in range(1, 16)]
    )
    return {
        a: {"name": f"User {a}", "country": "US", "kyc_risk": "low"}
        for a in accounts
    }


# ── Existing test stubs (now implemented) ─────────────────────────────────────

def test_known_pattern_chain(tiny_transactions: pl.DataFrame, tiny_accounts: pl.DataFrame) -> None:
    """
    In a 3-cycle (ACC001→ACC002→ACC003→ACC001):
    - ACC001 has both ACC002 and ACC003 as direct hop-1 counterparties
    - pattern_label is populated from the pattern map
    """
    graph = build_transaction_graph(tiny_transactions)
    overlay_map = {
        f"ACC{i:03d}": {"name": f"User {i}", "country": "US", "kyc_risk": "medium"}
        for i in range(1, 11)
    }
    pattern_map = build_pattern_map({"CYCLE": ["ACC001", "ACC002", "ACC003"]})

    chain = resolve_entity("ACC001", graph, overlay_map, pattern_map)

    assert "ACC002" in chain.hop1_counterparties
    assert "ACC003" in chain.hop1_counterparties
    assert chain.pattern_label == "CYCLE"


def test_isolated_account(tiny_accounts: pl.DataFrame) -> None:
    """Account with zero transactions returns empty hop lists without raising."""
    empty_df = pl.DataFrame({
        "from_account": pl.Series([], dtype=pl.Utf8),
        "to_account":   pl.Series([], dtype=pl.Utf8),
    })
    graph = build_transaction_graph(empty_df)
    overlay_map = {"ISOLATED": {"name": "Alice Smith", "country": "US", "kyc_risk": "low"}}

    chain = resolve_entity("ISOLATED", graph, overlay_map, {})

    assert chain.hop1_counterparties == []
    assert chain.hop2_counterparties == []
    assert chain.pattern_label is None


def test_hop2_cap(tiny_transactions: pl.DataFrame, tiny_accounts: pl.DataFrame) -> None:
    """Hub with 75 second-degree neighbours returns exactly HOP2_CAP=50 entries."""
    # Build a synthetic hub scenario independent of tiny_transactions
    from_accounts: list[str] = ["HUB"] * 5
    to_accounts: list[str] = [f"R{i:02d}" for i in range(1, 6)]
    for i in range(1, 6):
        for j in range(1, 16):
            from_accounts.append(f"R{i:02d}")
            to_accounts.append(f"R{i:02d}_D{j:02d}")
    hub_df = pl.DataFrame({"from_account": from_accounts, "to_account": to_accounts})

    graph = build_transaction_graph(hub_df)
    overlay_map = {"HUB": {"name": "Hub Account", "country": "US", "kyc_risk": "high"}}

    chain = resolve_entity("HUB", graph, overlay_map, {})

    assert len(chain.hop1_counterparties) == 5
    assert len(chain.hop2_counterparties) == HOP2_CAP
    assert chain.hop2_counterparties == sorted(chain.hop2_counterparties)


def test_cycle_no_infinite_loop(tiny_transactions: pl.DataFrame, tiny_accounts: pl.DataFrame) -> None:
    """Account in a 3-cycle terminates; no RecursionError or hang."""
    graph = build_transaction_graph(tiny_transactions)
    overlay_map = {"ACC001": {"name": "Cycle User", "country": "US", "kyc_risk": "low"}}

    chain = resolve_entity("ACC001", graph, overlay_map, {})

    # All three cycle nodes are direct neighbours of each other → hop2 is empty
    assert set(chain.hop1_counterparties) == {"ACC002", "ACC003"}
    assert chain.hop2_counterparties == []


# ── TestBuildTransactionGraph ─────────────────────────────────────────────────

class TestBuildTransactionGraph:
    def test_undirected_forward_edge(self, fanout_df: pl.DataFrame) -> None:
        g = build_transaction_graph(fanout_df)
        assert "R1" in g["HUB"]
        assert "HUB" in g["R1"]

    def test_undirected_reverse_edge(self, fanout_df: pl.DataFrame) -> None:
        g = build_transaction_graph(fanout_df)
        assert "R2" in g["LEAF"]
        assert "LEAF" in g["R2"]

    def test_self_loop_excluded(self) -> None:
        df = pl.DataFrame({"from_account": ["ACC001"], "to_account": ["ACC001"]})
        g = build_transaction_graph(df)
        assert "ACC001" not in g.get("ACC001", set())

    def test_duplicate_transactions_deduplicated(self) -> None:
        df = pl.DataFrame({
            "from_account": ["A", "A", "A"],
            "to_account":   ["B", "B", "B"],
        })
        g = build_transaction_graph(df)
        assert g["A"] == {"B"}
        assert g["B"] == {"A"}

    def test_empty_dataframe_returns_empty_graph(self) -> None:
        df = pl.DataFrame({
            "from_account": pl.Series([], dtype=pl.Utf8),
            "to_account":   pl.Series([], dtype=pl.Utf8),
        })
        g = build_transaction_graph(df)
        assert g == {}

    def test_account_ids_are_strings(self, fanout_df: pl.DataFrame) -> None:
        g = build_transaction_graph(fanout_df)
        for key in g:
            assert isinstance(key, str)
        for neighbours in g.values():
            for n in neighbours:
                assert isinstance(n, str)

    def test_3_cycle_each_has_two_neighbours(self, cycle_df: pl.DataFrame) -> None:
        g = build_transaction_graph(cycle_df)
        assert g["ACC001"] == {"ACC002", "ACC003"}
        assert g["ACC002"] == {"ACC001", "ACC003"}
        assert g["ACC003"] == {"ACC001", "ACC002"}

    def test_isolated_account_not_in_graph(self) -> None:
        """Account that never appears in transactions is absent from the graph."""
        df = pl.DataFrame({"from_account": ["A"], "to_account": ["B"]})
        g = build_transaction_graph(df)
        assert "PHANTOM" not in g


# ── TestBuildOverlayMap ───────────────────────────────────────────────────────

class TestBuildOverlayMap:
    def _write_parquet(self, tmp_path: Path, data: dict) -> Path:
        p = tmp_path / "overlay.parquet"
        pl.DataFrame(data).write_parquet(p)
        return p

    def test_known_account_profile_returned(self, tmp_path: Path) -> None:
        p = self._write_parquet(tmp_path, {
            "account_id": ["ACC001"],
            "name": ["Alice"],
            "country": ["US"],
            "kyc_risk": ["high"],
        })
        m = build_overlay_map(p)
        assert m["ACC001"]["name"] == "Alice"
        assert m["ACC001"]["country"] == "US"
        assert m["ACC001"]["kyc_risk"] == "high"

    def test_unknown_account_returns_none(self, tmp_path: Path) -> None:
        p = self._write_parquet(tmp_path, {
            "account_id": ["ACC001"],
            "name": ["Alice"],
            "country": ["US"],
            "kyc_risk": ["high"],
        })
        m = build_overlay_map(p)
        assert m.get("DOES_NOT_EXIST") is None

    def test_all_kyc_risk_values_preserved(self, tmp_path: Path) -> None:
        p = self._write_parquet(tmp_path, {
            "account_id": ["A", "B", "C"],
            "name": ["Alice", "Bob", "Carol"],
            "country": ["US", "UK", "CA"],
            "kyc_risk": ["low", "medium", "high"],
        })
        m = build_overlay_map(p)
        assert m["A"]["kyc_risk"] == "low"
        assert m["B"]["kyc_risk"] == "medium"
        assert m["C"]["kyc_risk"] == "high"

    def test_account_id_key_is_string(self, tmp_path: Path) -> None:
        p = self._write_parquet(tmp_path, {
            "account_id": ["ACC001"],
            "name": ["Alice"],
            "country": ["US"],
            "kyc_risk": ["low"],
        })
        m = build_overlay_map(p)
        for key in m:
            assert isinstance(key, str)


# ── TestBuildPatternMap ───────────────────────────────────────────────────────

class TestBuildPatternMap:
    def test_inversion_correct(self) -> None:
        patterns = {"FAN-OUT": ["A", "B"], "CYCLE": ["C"]}
        m = build_pattern_map(patterns)
        assert m["A"] == "FAN-OUT"
        assert m["B"] == "FAN-OUT"
        assert m["C"] == "CYCLE"

    def test_account_in_multiple_typologies_gets_alphabetically_first(self) -> None:
        # "BIPARTITE" < "STACK" alphabetically
        patterns = {"STACK": ["X"], "BIPARTITE": ["X"]}
        m = build_pattern_map(patterns)
        assert m["X"] == "BIPARTITE"

    def test_account_not_in_patterns_absent_from_map(self) -> None:
        m = build_pattern_map({"FAN-OUT": ["A"]})
        assert "Z" not in m

    def test_empty_patterns_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            m = build_pattern_map({})
        assert m == {}
        assert any("empty" in r.message.lower() for r in caplog.records)

    def test_deterministic_across_calls(self) -> None:
        patterns = {"Z_TYPE": ["A", "B"], "A_TYPE": ["A", "B"]}
        m1 = build_pattern_map(patterns)
        m2 = build_pattern_map(patterns)
        assert m1 == m2


# ── TestParsePatternFile ──────────────────────────────────────────────────────

class TestParsePatternFile:
    _SAMPLE = (
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT: Max 16-degree Fan-Out\n"
        "2022/09/01 00:06,021174,HUB001,012,RECV001,2848.96,Euro,2848.96,Euro,ACH,1\n"
        "2022/09/01 04:33,021174,HUB001,020,RECV002,8630.40,Euro,8630.40,Euro,ACH,1\n"
        "END LAUNDERING ATTEMPT - FAN-OUT\n"
        "\n"
        "BEGIN LAUNDERING ATTEMPT - CYCLE: Max 10 hops\n"
        "2022/09/02 10:00,001,CYCA,002,CYCB,500.00,USD,500.00,USD,Wire,1\n"
        "2022/09/02 11:00,002,CYCB,003,CYCC,500.00,USD,500.00,USD,Wire,1\n"
        "2022/09/02 12:00,003,CYCC,001,CYCA,500.00,USD,500.00,USD,Wire,1\n"
        "END LAUNDERING ATTEMPT - CYCLE\n"
    )

    def _write(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "patterns.txt"
        p.write_text(content)
        return p

    def test_typology_names_extracted(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, self._SAMPLE)
        result = parse_patterns_file(p)
        assert "FAN-OUT" in result
        assert "CYCLE" in result

    def test_from_accounts_in_result(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, self._SAMPLE)
        result = parse_patterns_file(p)
        assert "HUB001" in result["FAN-OUT"]

    def test_to_accounts_in_result(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, self._SAMPLE)
        result = parse_patterns_file(p)
        assert "RECV001" in result["FAN-OUT"]
        assert "RECV002" in result["FAN-OUT"]

    def test_cycle_accounts_all_present(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, self._SAMPLE)
        result = parse_patterns_file(p)
        cycle = result["CYCLE"]
        assert "CYCA" in cycle
        assert "CYCB" in cycle
        assert "CYCC" in cycle

    def test_no_duplicates_within_typology(self, tmp_path: Path) -> None:
        content = (
            "BEGIN LAUNDERING ATTEMPT - FAN-OUT:\n"
            "2022/01/01 00:00,001,HUB,002,RECV,100.0,USD,100.0,USD,ACH,1\n"
            "2022/01/02 00:00,001,HUB,002,RECV,200.0,USD,200.0,USD,ACH,1\n"
            "END LAUNDERING ATTEMPT - FAN-OUT\n"
        )
        p = self._write(tmp_path, content)
        result = parse_patterns_file(p)
        assert result["FAN-OUT"].count("HUB") == 1
        assert result["FAN-OUT"].count("RECV") == 1

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_patterns_file(tmp_path / "nonexistent.txt")

    def test_empty_file_returns_empty_dict(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, "")
        result = parse_patterns_file(p)
        assert result == {}

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        content = (
            "\n\nBEGIN LAUNDERING ATTEMPT - FAN-OUT:\n\n"
            "2022/01/01 00:00,001,HUB,002,RECV,1.0,USD,1.0,USD,ACH,1\n\n"
            "END LAUNDERING ATTEMPT - FAN-OUT\n\n"
        )
        p = self._write(tmp_path, content)
        result = parse_patterns_file(p)
        assert "FAN-OUT" in result


# ── TestResolveEntity ─────────────────────────────────────────────────────────

class TestResolveEntity:
    def test_hop1_direct_counterparties(self, fanout_df: pl.DataFrame,
                                        simple_overlay_map: dict) -> None:
        graph = build_transaction_graph(fanout_df)
        chain = resolve_entity("HUB", graph, simple_overlay_map, {})
        assert set(chain.hop1_counterparties) == {"R1", "R2", "R3"}

    def test_hop2_excludes_self(self, fanout_df: pl.DataFrame,
                                simple_overlay_map: dict) -> None:
        graph = build_transaction_graph(fanout_df)
        chain = resolve_entity("HUB", graph, simple_overlay_map, {})
        assert "HUB" not in chain.hop2_counterparties

    def test_hop2_excludes_hop1(self, fanout_df: pl.DataFrame,
                                simple_overlay_map: dict) -> None:
        graph = build_transaction_graph(fanout_df)
        chain = resolve_entity("HUB", graph, simple_overlay_map, {})
        hop1_set = set(chain.hop1_counterparties)
        assert not (hop1_set & set(chain.hop2_counterparties))

    def test_hop2_contains_leaf(self, fanout_df: pl.DataFrame,
                                simple_overlay_map: dict) -> None:
        graph = build_transaction_graph(fanout_df)
        chain = resolve_entity("HUB", graph, simple_overlay_map, {})
        assert "LEAF" in chain.hop2_counterparties

    def test_hop1_sorted(self, fanout_df: pl.DataFrame,
                         simple_overlay_map: dict) -> None:
        graph = build_transaction_graph(fanout_df)
        chain = resolve_entity("HUB", graph, simple_overlay_map, {})
        assert chain.hop1_counterparties == sorted(chain.hop1_counterparties)

    def test_hop2_sorted(self, hub_df: pl.DataFrame) -> None:
        graph = build_transaction_graph(hub_df)
        overlay_map = {"HUB": {"name": "Hub", "country": "US", "kyc_risk": "high"}}
        chain = resolve_entity("HUB", graph, overlay_map, {})
        assert chain.hop2_counterparties == sorted(chain.hop2_counterparties)

    def test_pattern_label_assigned(self, cycle_df: pl.DataFrame) -> None:
        graph = build_transaction_graph(cycle_df)
        overlay_map = {"ACC001": {"name": "Alice", "country": "US", "kyc_risk": "low"}}
        pattern_map = build_pattern_map({"CYCLE": ["ACC001"]})
        chain = resolve_entity("ACC001", graph, overlay_map, pattern_map)
        assert chain.pattern_label == "CYCLE"

    def test_pattern_label_none_for_unlabeled(self, cycle_df: pl.DataFrame) -> None:
        graph = build_transaction_graph(cycle_df)
        overlay_map = {"ACC002": {"name": "Bob", "country": "US", "kyc_risk": "low"}}
        pattern_map = build_pattern_map({"CYCLE": ["ACC001"]})  # ACC002 not in pattern
        chain = resolve_entity("ACC002", graph, overlay_map, pattern_map)
        assert chain.pattern_label is None

    def test_unknown_account_graceful(self, cycle_df: pl.DataFrame) -> None:
        graph = build_transaction_graph(cycle_df)
        chain = resolve_entity("PHANTOM", graph, {}, {})
        assert chain.name == "UNKNOWN"
        assert chain.country is None
        assert chain.kyc_risk is None

    def test_unknown_account_logs_warning(self, cycle_df: pl.DataFrame,
                                          caplog: pytest.LogCaptureFixture) -> None:
        graph = build_transaction_graph(cycle_df)
        with caplog.at_level(logging.WARNING):
            resolve_entity("PHANTOM", graph, {}, {})
        assert any("PHANTOM" in r.message for r in caplog.records)

    def test_name_from_overlay(self, cycle_df: pl.DataFrame) -> None:
        graph = build_transaction_graph(cycle_df)
        overlay_map = {"ACC001": {"name": "Test Name XYZ", "country": "US", "kyc_risk": "low"}}
        chain = resolve_entity("ACC001", graph, overlay_map, {})
        assert chain.name == "Test Name XYZ"

    def test_country_from_overlay(self, cycle_df: pl.DataFrame) -> None:
        graph = build_transaction_graph(cycle_df)
        overlay_map = {"ACC001": {"name": "Alice", "country": "DE", "kyc_risk": "medium"}}
        chain = resolve_entity("ACC001", graph, overlay_map, {})
        assert chain.country == "DE"

    def test_kyc_risk_from_overlay(self, cycle_df: pl.DataFrame) -> None:
        graph = build_transaction_graph(cycle_df)
        overlay_map = {"ACC001": {"name": "Alice", "country": "US", "kyc_risk": "high"}}
        chain = resolve_entity("ACC001", graph, overlay_map, {})
        assert chain.kyc_risk == "high"

    def test_return_type_is_entity_chain(self, cycle_df: pl.DataFrame) -> None:
        from aml_copilot.schemas import EntityChain
        graph = build_transaction_graph(cycle_df)
        overlay_map = {"ACC001": {"name": "Alice", "country": "US", "kyc_risk": "low"}}
        chain = resolve_entity("ACC001", graph, overlay_map, {})
        assert isinstance(chain, EntityChain)
        assert chain.account_id == "ACC001"

    def test_isolated_account_returns_empty_hops(self) -> None:
        graph: dict[str, set[str]] = {}
        overlay_map = {"ACC005": {"name": "Eve", "country": "FR", "kyc_risk": "low"}}
        chain = resolve_entity("ACC005", graph, overlay_map, {})
        assert chain.hop1_counterparties == []
        assert chain.hop2_counterparties == []

    def test_self_transfer_excluded_from_hops(self) -> None:
        """Account that only transacts with itself has empty hop lists."""
        df = pl.DataFrame({"from_account": ["ACC001"], "to_account": ["ACC001"]})
        graph = build_transaction_graph(df)
        overlay_map = {"ACC001": {"name": "Self Sender", "country": "US", "kyc_risk": "low"}}
        chain = resolve_entity("ACC001", graph, overlay_map, {})
        assert chain.hop1_counterparties == []
        assert chain.hop2_counterparties == []


# ── TestPatternChainIntegration ───────────────────────────────────────────────

@pytest.mark.integration
class TestPatternChainIntegration:
    _TRANS = Path("data/raw/HI-Small_Trans.csv")
    _PATTERNS = Path("data/raw/HI-Small_Patterns.txt")
    _OVERLAY = Path("data/processed/identity_overlay.parquet")

    # Known FAN-OUT hub from the first block of HI-Small_Patterns.txt
    _HUB_ACCOUNT = "800737690"

    @pytest.fixture(autouse=True)
    def require_data(self) -> None:
        if not self._TRANS.exists():
            pytest.skip(f"Transaction CSV not found: {self._TRANS}")
        if not self._PATTERNS.exists():
            pytest.skip(f"Patterns file not found: {self._PATTERNS}")
        if not self._OVERLAY.exists():
            pytest.skip(f"Overlay parquet not found: {self._OVERLAY}")

    @pytest.fixture
    def real_graph(self) -> dict[str, set[str]]:
        from aml_copilot.step0_scaffold.data_loader import load_transactions
        df = load_transactions(self._TRANS)
        return build_transaction_graph(df)

    @pytest.fixture
    def real_overlay_map(self) -> dict[str, dict]:
        return build_overlay_map(self._OVERLAY)

    @pytest.fixture
    def real_pattern_map(self) -> dict[str, str]:
        patterns = parse_patterns_file(self._PATTERNS)
        return build_pattern_map(patterns)

    def test_parse_returns_all_8_typologies(self) -> None:
        patterns = parse_patterns_file(self._PATTERNS)
        expected = {"FAN-OUT", "FAN-IN", "CYCLE", "BIPARTITE",
                    "STACK", "RANDOM", "SCATTER-GATHER", "GATHER-SCATTER"}
        assert expected <= set(patterns.keys()), (
            f"Missing typologies: {expected - set(patterns.keys())}"
        )

    def test_pattern_map_nonempty(self, real_pattern_map: dict) -> None:
        assert len(real_pattern_map) > 0

    def test_hub_account_in_pattern_map(self, real_pattern_map: dict) -> None:
        assert self._HUB_ACCOUNT in real_pattern_map

    def test_hub_pattern_label_is_fanout(self, real_pattern_map: dict) -> None:
        assert real_pattern_map[self._HUB_ACCOUNT] == "FAN-OUT"

    def test_known_fanout_hub_has_hop1_counterparties(
        self,
        real_graph: dict,
        real_overlay_map: dict,
        real_pattern_map: dict,
    ) -> None:
        chain = resolve_entity(
            self._HUB_ACCOUNT, real_graph, real_overlay_map, real_pattern_map
        )
        assert len(chain.hop1_counterparties) > 0, (
            f"Hub {self._HUB_ACCOUNT!r} has no hop-1 counterparties — "
            "graph may not have loaded correctly"
        )

    def test_known_fanout_hub_pattern_label_set(
        self,
        real_graph: dict,
        real_overlay_map: dict,
        real_pattern_map: dict,
    ) -> None:
        chain = resolve_entity(
            self._HUB_ACCOUNT, real_graph, real_overlay_map, real_pattern_map
        )
        assert chain.pattern_label == "FAN-OUT"

    def test_hub_hop2_capped(
        self,
        real_graph: dict,
        real_overlay_map: dict,
        real_pattern_map: dict,
    ) -> None:
        chain = resolve_entity(
            self._HUB_ACCOUNT, real_graph, real_overlay_map, real_pattern_map
        )
        assert len(chain.hop2_counterparties) <= HOP2_CAP

    def test_undirected_invariant_holds(self, real_graph: dict) -> None:
        """Every hop-1 counterparty of the hub must also have the hub in its adjacency."""
        hub_neighbours = real_graph.get(self._HUB_ACCOUNT, set())
        assert len(hub_neighbours) > 0
        for acct in hub_neighbours:
            assert self._HUB_ACCOUNT in real_graph.get(acct, set()), (
                f"Undirected invariant violated: {acct!r} in hub adjacency "
                f"but hub not in {acct!r} adjacency"
            )

    def test_overlay_row_count(self, real_overlay_map: dict) -> None:
        assert len(real_overlay_map) == 515_080

    def test_name_field_is_not_unknown_for_hub(
        self,
        real_graph: dict,
        real_overlay_map: dict,
        real_pattern_map: dict,
    ) -> None:
        chain = resolve_entity(
            self._HUB_ACCOUNT, real_graph, real_overlay_map, real_pattern_map
        )
        assert chain.name != "UNKNOWN", (
            f"Hub {self._HUB_ACCOUNT!r} not found in overlay — "
            "overlay_map or account ID format mismatch"
        )
