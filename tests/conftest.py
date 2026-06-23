"""
Shared pytest fixtures.

Step 1 fixtures (mock_ofac_records, tiny_accounts) are self-contained and
require no disk I/O. Integration tests that need the real OFAC files or
full accounts.parquet are marked @pytest.mark.integration and auto-skip
when the files are absent.
"""
from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from aml_copilot.schemas import AnomalyScore, EvalCase, RuleFiring, SanctionsHit
from aml_copilot.step1_identity.ofac_reader import OFACRecord
from aml_copilot.step2_sanctions.index import OFACEntry, OFACIndex, build_ofac_index


# ── Step 0 fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def tiny_transactions() -> pl.DataFrame:
    """Minimal synthetic transaction DataFrame for unit tests (no disk I/O)."""
    return pl.DataFrame({
        "timestamp": [datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 1, 3)],
        "from_bank": ["BankA", "BankB", "BankC"],
        "from_account": ["ACC001", "ACC002", "ACC003"],
        "to_bank": ["BankB", "BankC", "BankA"],
        "to_account": ["ACC002", "ACC003", "ACC001"],
        "amount_received": [100.0, 200.0, 300.0],
        "receiving_currency": ["USD", "USD", "EUR"],
        "amount_paid": [100.0, 200.0, 300.0],
        "payment_currency": ["USD", "USD", "EUR"],
        "payment_format": ["Wire", "Wire", "ACH"],
        "is_laundering": pl.Series([0, 0, 1], dtype=pl.Int8),
    })


@pytest.fixture
def tiny_accounts() -> pl.DataFrame:
    """Minimal synthetic accounts DataFrame (10 rows) for Step 1 unit tests."""
    return pl.DataFrame({
        "account_id": [f"ACC{i:03d}" for i in range(1, 11)],
    })


# ── Step 1 fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_ofac_records() -> list[OFACRecord]:
    """
    30 mock OFACRecord entries covering all TP flavors and HN targets.
    Names are chosen so the overlay builder can find valid candidates for each flavor.

    Exact (5): entries with non-ASCII diacritics where NFKD produces a different string.
    Transliteration (5): entries with substitutable romanization patterns.
    Typo/OCR (5): entries suitable for character-level OCR mutations.
    Partial reorder (5): multi-token entries.
    HN targets (15+): entries with 5+ char surnames for surname-sharing HNs.
    """
    return [
        # ── Exact flavor targets (non-ASCII; NFKD produces a different string) ──
        OFACRecord(uid="E001", canonical_name="García López", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="E002", canonical_name="Müller Heinz", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="E003", canonical_name="Ñoño Ramírez", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="E004", canonical_name="Fàtimàh Hüseyn", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="E005", canonical_name="Björk Ström", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="E006", canonical_name="Pérèz Sánchez", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="E007", canonical_name="Ünlü Çelik", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="E008", canonical_name="Köhler Jürgen", entry_type="Individual", list_name="SDN", aka_names=[]),
        # ── Transliteration flavor targets ───────────────────────────────────
        OFACRecord(uid="T001", canonical_name="Mohammed Hassan", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="T002", canonical_name="Abdullah Hussain", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="T003", canonical_name="Ahmad Khalid", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="T004", canonical_name="Usama Ibrahim", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="T005", canonical_name="Mukhtar Soliman", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="T006", canonical_name="Khoury Michel", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="T007", canonical_name="Mahmoud Khalil", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="T008", canonical_name="Youssef Philippe", entry_type="Individual", list_name="SDN", aka_names=[]),
        # ── Typo/OCR flavor targets ───────────────────────────────────────────
        OFACRecord(uid="O001", canonical_name="Roberto Garcia", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="O002", canonical_name="Hassan Kamil", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="O003", canonical_name="Victor Sokolov", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="O004", canonical_name="Boris Petrov", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="O005", canonical_name="Carlos Mendez", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="O006", canonical_name="Andrei Volkov", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="O007", canonical_name="George Orlov", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="O008", canonical_name="Sergei Morozov", entry_type="Individual", list_name="SDN", aka_names=[]),
        # ── Partial reorder flavor targets (multi-token names) ────────────────
        OFACRecord(uid="R001", canonical_name="John William Smith", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="R002", canonical_name="Carlos Eduardo Fernandez", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="R003", canonical_name="Ali Hassan Ibrahim", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="R004", canonical_name="Ivan Mikhail Petrov", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="R005", canonical_name="Omar Abdullah Khalid", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="R006", canonical_name="Maria Elena Gonzalez", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="R007", canonical_name="David James Wilson", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="R008", canonical_name="Fatima Ahmed Nasser", entry_type="Individual", list_name="SDN", aka_names=[]),
        # ── HN targets (surname ≥5 chars for reliable JW band) ───────────────
        OFACRecord(uid="H001", canonical_name="Ibrahim Hassan", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H002", canonical_name="Abdul Rahman", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H003", canonical_name="Yusuf Adnan", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H004", canonical_name="Majid Karimi", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H005", canonical_name="Tariq Waheed", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H006", canonical_name="Dmitri Volkov", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H007", canonical_name="Anton Petrov", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H008", canonical_name="Sergio Reyes", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H009", canonical_name="Khalid Mansour", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H010", canonical_name="Samir Haddad", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H011", canonical_name="Nasser Fahad", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H012", canonical_name="Bashir Rajab", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H013", canonical_name="Faisal Dabbagh", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H014", canonical_name="Walid Hamdan", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H015", canonical_name="Rafiq Saleem", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H016", canonical_name="Karim Diallo", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H017", canonical_name="Pavel Novak", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H018", canonical_name="Alexei Smirnov", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H019", canonical_name="Diego Vargas", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H020", canonical_name="Hector Morales", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H021", canonical_name="Nguyen Minh", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H022", canonical_name="Anwar Sadat", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H023", canonical_name="Hamza Sharif", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H024", canonical_name="Rashid Qureshi", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H025", canonical_name="Usman Farooq", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H026", canonical_name="Mirza Baig", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H027", canonical_name="Zafar Iqbal", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H028", canonical_name="Tariq Javed", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H029", canonical_name="Hassan Malik", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H030", canonical_name="Bilal Chaudhry", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H031", canonical_name="Omar Farouk", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H032", canonical_name="Abdul Aziz", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H033", canonical_name="Jamal Bakri", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H034", canonical_name="Zakaria Moussa", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H035", canonical_name="Mustafa Yilmaz", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H036", canonical_name="Osman Karahan", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H037", canonical_name="Ahmet Demir", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H038", canonical_name="Mehmet Kaya", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H039", canonical_name="Kemal Ozturk", entry_type="Individual", list_name="SDN", aka_names=[]),
        OFACRecord(uid="H040", canonical_name="Selim Arslan", entry_type="Individual", list_name="SDN", aka_names=[]),
    ]


# ── Step 2 fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_ofac_index() -> OFACIndex:
    """Small OFACIndex for Step 2 unit tests (built via build_ofac_index)."""
    records = [
        OFACRecord(uid="1", canonical_name="Jose Garcia", entry_type="Individual",
                   list_name="SDN", aka_names=["Garcia Jose"]),
        OFACRecord(uid="2", canonical_name="Hassan Ibrahim", entry_type="Individual",
                   list_name="SDN", aka_names=["Ibrahim Hassan"]),
    ]
    return build_ofac_index(records)


# ── Other shared fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def sample_sanctions_hit() -> SanctionsHit:
    return SanctionsHit(
        account_id="ACC001",
        assigned_name="Jose Garcia",
        ofac_uid="1",
        list_source="SDN",
        match_score=0.95,
        scorer_used="jaro_winkler",
        matched_name_type="canonical",
    )


@pytest.fixture
def sample_rule_firing() -> RuleFiring:
    return RuleFiring(
        rule_id="STRUCT_001",
        severity=3,
        account_id="ACC001",
        evidence={"txn_count": 4},
        window_start=datetime(2022, 1, 1),
        window_end=datetime(2022, 1, 2),
    )


@pytest.fixture
def sample_anomaly_score() -> AnomalyScore:
    return AnomalyScore(
        account_id="ACC001",
        score=-0.3,
        percentile=0.95,
        is_flagged=True,
        excluded_features=["balance_delta", "net_flow"],
    )


@pytest.fixture
def sample_eval_case() -> EvalCase:
    return EvalCase(
        case_id="CASE_001",
        account_id="ACC001",
        gold_label="ESCALATE",
        case_type="sanctions_hit",
        relevant_txn_ids=["TXN001", "TXN002"],
        notes="Mock eval case",
    )
