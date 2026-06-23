"""
Unit tests for Step 1: identity overlay and ground-truth construction.

All tests use mock OFAC records and tiny_accounts (no disk I/O).
Integration tests marked @pytest.mark.integration require real OFAC files
and are skipped automatically when absent.
"""
from __future__ import annotations

import csv
import random
import tempfile
from pathlib import Path

import polars as pl
import pytest

from aml_copilot.step1_identity.ofac_reader import OFACRecord, build_raw_name_set
from aml_copilot.step1_identity.overlay import (
    FAKER_SEED,
    _gen_exact,
    _gen_transliteration,
    _gen_typo_ocr,
    _gen_partial_reorder,
    _gen_hard_negative,
    build_identity_overlay,
    save_overlay,
)
from aml_copilot.step1_identity.ground_truth import save_ground_truth, load_ground_truth
from aml_copilot.utils.normalize import nfkd_normalize, normalize_name, score_names
from aml_copilot.utils.checksum import compute_sha256, append_checksum, verify_checksums


# ── Normalize utilities ───────────────────────────────────────────────────────

class TestNfkdNormalize:
    def test_removes_diacritics(self) -> None:
        assert nfkd_normalize("García") == "Garcia"
        assert nfkd_normalize("Müller") == "Muller"
        assert nfkd_normalize("Ñoño") == "Nono"

    def test_pure_ascii_unchanged(self) -> None:
        assert nfkd_normalize("John Smith") == "John Smith"

    def test_empty_string(self) -> None:
        assert nfkd_normalize("") == ""

    def test_multiple_diacritics(self) -> None:
        assert nfkd_normalize("Björk Ström") == "Bjork Strom"


class TestScoreNames:
    def test_identical_names_score_1(self) -> None:
        assert score_names("John Smith", "John Smith") == pytest.approx(1.0)

    def test_reversed_tokens_score_high(self) -> None:
        # token_sort_ratio handles reordering
        assert score_names("Smith John", "John Smith") >= 0.90

    def test_clearly_different_names_score_low(self) -> None:
        assert score_names("John Smith", "Xuan Pham") < 0.70

    def test_normalized_diacritics_score_1(self) -> None:
        # nfkd(García) == Garcia → score against normalized OFAC entry = 1.0
        assert score_names("Garcia", "García") == pytest.approx(1.0)


# ── Checksum utilities ────────────────────────────────────────────────────────

class TestChecksum:
    def test_compute_sha256_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("hello")
        d1 = compute_sha256(f)
        d2 = compute_sha256(f)
        assert d1 == d2
        assert len(d1) == 64

    def test_append_checksum_write_once(self, tmp_path: Path) -> None:
        f = tmp_path / "artifact.csv"
        f.write_text("data")
        cs = tmp_path / "checksums.sha256"
        append_checksum(f, cs)
        with pytest.raises(RuntimeError, match="frozen"):
            append_checksum(f, cs)

    def test_verify_passes_on_unchanged_file(self, tmp_path: Path) -> None:
        f = tmp_path / "artifact.csv"
        f.write_text("data")
        cs = tmp_path / "checksums.sha256"
        append_checksum(f, cs)
        verify_checksums(cs)  # must not raise

    def test_verify_fails_on_modified_file(self, tmp_path: Path) -> None:
        f = tmp_path / "artifact.csv"
        f.write_text("original")
        cs = tmp_path / "checksums.sha256"
        append_checksum(f, cs)
        f.write_text("tampered")
        with pytest.raises(RuntimeError, match="mismatch"):
            verify_checksums(cs)

    def test_verify_silent_when_no_checksum_file(self, tmp_path: Path) -> None:
        verify_checksums(tmp_path / "nonexistent.sha256")  # must not raise


# ── TP name generators ────────────────────────────────────────────────────────

class TestGenExact:
    def test_produces_nfkd_form(self) -> None:
        entry = OFACRecord(uid="1", canonical_name="García López",
                           entry_type="Individual", list_name="SDN", aka_names=[])
        raw_names: set[str] = set()
        name = _gen_exact(entry, raw_names)
        assert name == "Garcia Lopez"

    def test_returns_none_for_pure_ascii(self) -> None:
        entry = OFACRecord(uid="1", canonical_name="John Smith",
                           entry_type="Individual", list_name="SDN", aka_names=[])
        assert _gen_exact(entry, set()) is None

    def test_returns_none_if_normalized_in_raw_names(self) -> None:
        entry = OFACRecord(uid="1", canonical_name="García López",
                           entry_type="Individual", list_name="SDN", aka_names=[])
        raw_names = {"Garcia Lopez"}  # normalized form already exists as another raw name
        assert _gen_exact(entry, raw_names) is None

    def test_result_not_in_raw_names(self) -> None:
        entry = OFACRecord(uid="1", canonical_name="Müller Heinz",
                           entry_type="Individual", list_name="SDN", aka_names=[])
        name = _gen_exact(entry, set())
        assert name not in {"Müller Heinz"}  # must differ from raw canonical

    def test_score_is_1_against_canonical(self) -> None:
        entry = OFACRecord(uid="1", canonical_name="Ñoño Ramírez",
                           entry_type="Individual", list_name="SDN", aka_names=[])
        name = _gen_exact(entry, set())
        assert name is not None
        assert score_names(name, entry.canonical_name) == pytest.approx(1.0)


class TestGenTransliteration:
    @pytest.fixture
    def rng(self) -> random.Random:
        return random.Random(42)

    def test_score_at_least_090(self, rng: random.Random) -> None:
        entry = OFACRecord(uid="T001", canonical_name="Mohammed Hassan",
                           entry_type="Individual", list_name="SDN", aka_names=[])
        name = _gen_transliteration(entry, set(), rng)
        assert name is not None, "transliteration generator returned None"
        assert score_names(name, entry.canonical_name) >= 0.90

    def test_differs_from_canonical(self, rng: random.Random) -> None:
        entry = OFACRecord(uid="T001", canonical_name="Mohammed Hassan",
                           entry_type="Individual", list_name="SDN", aka_names=[])
        name = _gen_transliteration(entry, set(), rng)
        assert name is not None
        assert normalize_name(name) != normalize_name(entry.canonical_name)

    def test_not_in_raw_names(self, rng: random.Random) -> None:
        entry = OFACRecord(uid="T001", canonical_name="Mohammed Hassan",
                           entry_type="Individual", list_name="SDN", aka_names=[])
        raw = {"Mohammed Hassan"}
        name = _gen_transliteration(entry, raw, rng)
        if name is not None:
            assert name not in raw


class TestGenTypoOcr:
    @pytest.fixture
    def rng(self) -> random.Random:
        return random.Random(42)

    def test_score_at_least_085(self, rng: random.Random) -> None:
        entry = OFACRecord(uid="O001", canonical_name="Roberto Garcia",
                           entry_type="Individual", list_name="SDN", aka_names=[])
        name = _gen_typo_ocr(entry, set(), rng)
        assert name is not None, "typo_ocr generator returned None"
        assert score_names(name, entry.canonical_name) >= 0.85

    def test_differs_from_canonical(self, rng: random.Random) -> None:
        entry = OFACRecord(uid="O001", canonical_name="Roberto Garcia",
                           entry_type="Individual", list_name="SDN", aka_names=[])
        name = _gen_typo_ocr(entry, set(), rng)
        assert name is not None
        assert normalize_name(name) != normalize_name(entry.canonical_name)


class TestGenPartialReorder:
    def test_reversed_tokens(self) -> None:
        entry = OFACRecord(uid="R001", canonical_name="John William Smith",
                           entry_type="Individual", list_name="SDN", aka_names=[])
        name = _gen_partial_reorder(entry, set())
        assert name is not None
        tokens_original = nfkd_normalize(entry.canonical_name).split()
        tokens_result = name.split()
        assert sorted(tokens_original) == sorted([t.lower() for t in tokens_result]) or \
               sorted(t.lower() for t in tokens_original) == sorted(t.lower() for t in tokens_result)

    def test_token_sort_ratio_above_090(self) -> None:
        from rapidfuzz import fuzz
        entry = OFACRecord(uid="R001", canonical_name="John William Smith",
                           entry_type="Individual", list_name="SDN", aka_names=[])
        name = _gen_partial_reorder(entry, set())
        assert name is not None
        tsr = fuzz.token_sort_ratio(normalize_name(name), normalize_name(entry.canonical_name)) / 100.0
        assert tsr >= 0.90

    def test_returns_none_for_single_token(self) -> None:
        entry = OFACRecord(uid="R001", canonical_name="Anonymized",
                           entry_type="Individual", list_name="SDN", aka_names=[])
        assert _gen_partial_reorder(entry, set()) is None

    def test_returns_none_if_reversed_name_in_raw_names(self) -> None:
        entry = OFACRecord(uid="R001", canonical_name="John Smith",
                           entry_type="Individual", list_name="SDN", aka_names=[])
        assert _gen_partial_reorder(entry, {"Smith John"}) is None


# ── HN generator ─────────────────────────────────────────────────────────────

class TestGenHardNegative:
    @pytest.fixture
    def rng(self) -> random.Random:
        return random.Random(42)

    def test_target_score_in_band(
        self, rng: random.Random, mock_ofac_records: list[OFACRecord]
    ) -> None:
        from faker import Faker
        Faker.seed(42)
        fake = Faker()
        target = next(r for r in mock_ofac_records if r.uid == "H001")
        name = _gen_hard_negative(target, mock_ofac_records, fake, rng)
        assert name is not None, "HN generator returned None for H001"
        s = score_names(name, target.canonical_name)
        assert 0.80 <= s <= 0.88, f"HN score {s:.4f} outside [0.80, 0.88] for '{name}'"

    def test_max_score_below_threshold(
        self, rng: random.Random, mock_ofac_records: list[OFACRecord]
    ) -> None:
        from faker import Faker
        Faker.seed(42)
        fake = Faker()
        target = next(r for r in mock_ofac_records if r.uid == "H001")
        name = _gen_hard_negative(target, mock_ofac_records, fake, rng)
        assert name is not None
        from aml_copilot.step1_identity.overlay import _max_score_against_records
        max_s, _ = _max_score_against_records(name, mock_ofac_records)
        assert max_s < 0.90, f"HN max_score {max_s:.4f} >= 0.90 for '{name}'"

    def test_differs_from_target(
        self, rng: random.Random, mock_ofac_records: list[OFACRecord]
    ) -> None:
        from faker import Faker
        Faker.seed(42)
        fake = Faker()
        target = next(r for r in mock_ofac_records if r.uid == "H001")
        name = _gen_hard_negative(target, mock_ofac_records, fake, rng)
        assert name is not None
        assert normalize_name(name) != normalize_name(target.canonical_name)

    def test_returns_none_for_short_surname(
        self, rng: random.Random, mock_ofac_records: list[OFACRecord]
    ) -> None:
        from faker import Faker
        Faker.seed(42)
        fake = Faker()
        # Entry with very short surname
        entry = OFACRecord(uid="X", canonical_name="Ali Wu",
                           entry_type="Individual", list_name="SDN", aka_names=[])
        result = _gen_hard_negative(entry, mock_ofac_records, fake, rng)
        # "Wu" is 2 chars — should return None due to short surname guard
        assert result is None


# ── Full overlay builder ──────────────────────────────────────────────────────

class TestBuildIdentityOverlay:
    """
    These tests use a larger mock accounts fixture and the shared mock_ofac_records.
    The build is seeded so results are deterministic.
    """

    @pytest.fixture
    def accounts_parquet(self, tmp_path: Path) -> Path:
        """Write a 100-row synthetic accounts.parquet for overlay tests."""
        df = pl.DataFrame({"account_id": [f"ACC{i:04d}" for i in range(100)]})
        p = tmp_path / "accounts.parquet"
        df.write_parquet(p)
        return p

    def test_overlay_row_count(
        self, accounts_parquet: Path, mock_ofac_records: list[OFACRecord]
    ) -> None:
        overlay, _ = build_identity_overlay(accounts_parquet, mock_ofac_records, seed=42)
        assert len(overlay) == 100

    def test_overlay_account_id_unique(
        self, accounts_parquet: Path, mock_ofac_records: list[OFACRecord]
    ) -> None:
        overlay, _ = build_identity_overlay(accounts_parquet, mock_ofac_records, seed=42)
        assert overlay["account_id"].n_unique() == len(overlay)

    def test_overlay_columns(
        self, accounts_parquet: Path, mock_ofac_records: list[OFACRecord]
    ) -> None:
        overlay, _ = build_identity_overlay(accounts_parquet, mock_ofac_records, seed=42)
        assert set(overlay.columns) == {"account_id", "name", "country", "kyc_risk"}

    def test_no_raw_ofac_name_in_overlay(
        self, accounts_parquet: Path, mock_ofac_records: list[OFACRecord]
    ) -> None:
        overlay, _ = build_identity_overlay(accounts_parquet, mock_ofac_records, seed=42)
        raw_names = build_raw_name_set(mock_ofac_records)
        overlay_names = set(overlay["name"].to_list())
        leaked = overlay_names & raw_names
        assert not leaked, (
            f"{len(leaked)} raw OFAC name(s) found in overlay — "
            f"affected account_ids visible in overlay column only"
        )

    def test_fixture_rows_count(
        self, accounts_parquet: Path, mock_ofac_records: list[OFACRecord]
    ) -> None:
        _, rows = build_identity_overlay(accounts_parquet, mock_ofac_records, seed=42)
        assert len(rows) == 50

    def test_exactly_20_positives(
        self, accounts_parquet: Path, mock_ofac_records: list[OFACRecord]
    ) -> None:
        _, rows = build_identity_overlay(accounts_parquet, mock_ofac_records, seed=42)
        assert sum(1 for r in rows if r["gold_is_match"]) == 20

    def test_exactly_30_hard_negatives(
        self, accounts_parquet: Path, mock_ofac_records: list[OFACRecord]
    ) -> None:
        _, rows = build_identity_overlay(accounts_parquet, mock_ofac_records, seed=42)
        assert sum(1 for r in rows if not r["gold_is_match"]) == 30

    def test_flavor_distribution(
        self, accounts_parquet: Path, mock_ofac_records: list[OFACRecord]
    ) -> None:
        _, rows = build_identity_overlay(accounts_parquet, mock_ofac_records, seed=42)
        from collections import Counter
        counts = Counter(r["match_flavor"] for r in rows)
        assert counts["exact"] == 5
        assert counts["transliteration"] == 5
        assert counts["typo_ocr"] == 5
        assert counts["partial_reorder"] == 5
        assert counts["hard_negative"] == 30

    def test_tp_score_within_bounds(
        self, accounts_parquet: Path, mock_ofac_records: list[OFACRecord]
    ) -> None:
        _, rows = build_identity_overlay(accounts_parquet, mock_ofac_records, seed=42)
        uid_to_entry = {r.uid: r for r in mock_ofac_records}
        for row in rows:
            if not row["gold_is_match"]:
                continue
            entry = uid_to_entry[row["ofac_uid"]]
            s = score_names(row["assigned_name"], entry.canonical_name)
            assert s >= row["expected_score_min"], (
                f"TP {row['match_flavor']} uid={row['ofac_uid']}: "
                f"score {s:.4f} < min {row['expected_score_min']}"
            )

    def test_hard_negative_target_score_in_band(
        self, accounts_parquet: Path, mock_ofac_records: list[OFACRecord]
    ) -> None:
        _, rows = build_identity_overlay(accounts_parquet, mock_ofac_records, seed=42)
        uid_to_entry = {r.uid: r for r in mock_ofac_records}
        for row in rows:
            if row["gold_is_match"]:
                continue
            entry = uid_to_entry[row["ofac_uid"]]
            s = score_names(row["assigned_name"], entry.canonical_name)
            assert 0.80 <= s <= 0.88, (
                f"HN uid={row['ofac_uid']} name='{row['assigned_name']}': "
                f"score {s:.4f} not in [0.80, 0.88]"
            )

    def test_hard_negative_max_score_below_threshold(
        self, accounts_parquet: Path, mock_ofac_records: list[OFACRecord]
    ) -> None:
        """Full-index check: no HN scores ≥ 0.90 against any OFAC entry."""
        from aml_copilot.step1_identity.overlay import _max_score_against_records
        _, rows = build_identity_overlay(accounts_parquet, mock_ofac_records, seed=42)
        for row in rows:
            if row["gold_is_match"]:
                continue
            max_s, _ = _max_score_against_records(row["assigned_name"], mock_ofac_records)
            assert max_s < 0.90, (
                f"HN '{row['assigned_name']}' scores {max_s:.4f} against full index — "
                f"would be a false TP hit"
            )

    def test_faker_seed_reproducibility(
        self, accounts_parquet: Path, mock_ofac_records: list[OFACRecord]
    ) -> None:
        overlay1, rows1 = build_identity_overlay(accounts_parquet, mock_ofac_records, seed=42)
        overlay2, rows2 = build_identity_overlay(accounts_parquet, mock_ofac_records, seed=42)
        assert overlay1.equals(overlay2), "Overlay is not deterministic with same seed"
        names1 = [r["assigned_name"] for r in rows1]
        names2 = [r["assigned_name"] for r in rows2]
        assert names1 == names2, "Ground truth names differ between runs with same seed"

    def test_different_seed_produces_different_output(
        self, accounts_parquet: Path, mock_ofac_records: list[OFACRecord]
    ) -> None:
        overlay1, _ = build_identity_overlay(accounts_parquet, mock_ofac_records, seed=42)
        overlay2, _ = build_identity_overlay(accounts_parquet, mock_ofac_records, seed=99)
        assert not overlay1.equals(overlay2)


# ── save_overlay ──────────────────────────────────────────────────────────────

class TestSaveOverlay:
    def test_writes_parquet(self, tmp_path: Path) -> None:
        df = pl.DataFrame({"account_id": ["A"], "name": ["N"], "country": ["US"], "kyc_risk": ["low"]})
        p = tmp_path / "overlay.parquet"
        save_overlay(df, p)
        assert p.exists()
        assert len(pl.read_parquet(p)) == 1


# ── Ground truth freeze ───────────────────────────────────────────────────────

class TestGroundTruthFreeze:
    def _make_rows(self) -> list[dict]:
        rows = []
        flavors = ["exact", "transliteration", "typo_ocr", "partial_reorder"]
        row_id = 0
        for flavor in flavors:
            for i in range(5):
                min_s = 1.0 if flavor == "exact" else 0.90 if flavor != "typo_ocr" else 0.85
                rows.append({
                    "row_id": row_id,
                    "account_id": f"ACC{row_id:03d}",
                    "assigned_name": f"Test Name {row_id}",
                    "ofac_uid": f"TP{row_id:03d}",
                    "ofac_canonical_name": f"Canonical {row_id}",
                    "match_flavor": flavor,
                    "expected_score_min": min_s,
                    "expected_score_max": 1.0,
                    "gold_is_match": True,
                })
                row_id += 1
        for i in range(30):
            rows.append({
                "row_id": row_id,
                "account_id": f"ACC{row_id:03d}",
                "assigned_name": f"Hard Neg {i}",
                "ofac_uid": f"HN{i:03d}",
                "ofac_canonical_name": f"Target {i}",
                "match_flavor": "hard_negative",
                "expected_score_min": 0.80,
                "expected_score_max": 0.88,
                "gold_is_match": False,
            })
            row_id += 1
        return rows

    def test_writes_50_rows(self, tmp_path: Path) -> None:
        rows = self._make_rows()
        p = tmp_path / "ground_truth_matches.csv"
        cs = tmp_path / "checksums.sha256"
        save_ground_truth(rows, p, cs)
        loaded = load_ground_truth(p)
        assert len(loaded) == 50

    def test_write_once_raises_on_second_call(self, tmp_path: Path) -> None:
        rows = self._make_rows()
        p = tmp_path / "ground_truth_matches.csv"
        cs = tmp_path / "checksums.sha256"
        save_ground_truth(rows, p, cs)
        with pytest.raises(RuntimeError, match="frozen"):
            save_ground_truth(rows, p, cs)

    def test_checksum_written_after_save(self, tmp_path: Path) -> None:
        rows = self._make_rows()
        p = tmp_path / "ground_truth_matches.csv"
        cs = tmp_path / "checksums.sha256"
        save_ground_truth(rows, p, cs)
        assert cs.exists()
        content = cs.read_text()
        assert "ground_truth_matches.csv" in content

    def test_file_unchanged_after_freeze(self, tmp_path: Path) -> None:
        rows = self._make_rows()
        p = tmp_path / "ground_truth_matches.csv"
        cs = tmp_path / "checksums.sha256"
        save_ground_truth(rows, p, cs)
        digest_before = compute_sha256(p)
        # Second call raises before touching the file
        try:
            save_ground_truth(rows, p, cs)
        except RuntimeError:
            pass
        assert compute_sha256(p) == digest_before

    def test_validates_row_count(self, tmp_path: Path) -> None:
        rows = self._make_rows()[:10]  # only 10, not 50
        p = tmp_path / "ground_truth_matches.csv"
        cs = tmp_path / "checksums.sha256"
        with pytest.raises(ValueError, match="50"):
            save_ground_truth(rows, p, cs)

    def test_load_round_trips_correctly(self, tmp_path: Path) -> None:
        rows = self._make_rows()
        p = tmp_path / "ground_truth_matches.csv"
        cs = tmp_path / "checksums.sha256"
        save_ground_truth(rows, p, cs)
        loaded = load_ground_truth(p)
        assert loaded[0].match_flavor == "exact"
        assert loaded[0].gold_is_match is True
        assert loaded[20].match_flavor == "hard_negative"
        assert loaded[20].gold_is_match is False

    def test_ofac_canonical_name_not_in_log_output(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Raw OFAC canonical names must not appear in INFO-level log output."""
        import logging
        rows = self._make_rows()
        p = tmp_path / "ground_truth_matches.csv"
        cs = tmp_path / "checksums.sha256"
        with caplog.at_level(logging.INFO):
            save_ground_truth(rows, p, cs)
        canonical_names = {r["ofac_canonical_name"] for r in rows}
        for record in caplog.records:
            assert not any(cn in record.message for cn in canonical_names), (
                f"OFAC canonical name leaked into INFO log: '{record.message}'"
            )


# ── OFAC reader ───────────────────────────────────────────────────────────────

class TestBuildRawNameSet:
    def test_includes_canonical_and_akas(self) -> None:
        records = [
            OFACRecord(uid="1", canonical_name="Hassan Ibrahim",
                       entry_type="Individual", list_name="SDN",
                       aka_names=["Ibrahim Hassan", "H. Ibrahim"]),
        ]
        names = build_raw_name_set(records)
        assert "Hassan Ibrahim" in names
        assert "Ibrahim Hassan" in names
        assert "H. Ibrahim" in names

    def test_empty_records(self) -> None:
        assert build_raw_name_set([]) == set()


# ── OFAC parser sanity (requires real XML) ────────────────────────────────────

@pytest.mark.integration
class TestOFACParserSanity:
    """Smoke tests that the Advanced XML v3 parser produces non-empty output."""

    _SDN = Path("data/raw/ofac/sdn_advanced.xml")

    @pytest.fixture(autouse=True)
    def require_sdn(self) -> None:
        if not self._SDN.exists():
            pytest.skip(f"OFAC SDN file not found at {self._SDN}")

    def _records(self):
        from aml_copilot.step1_identity.ofac_reader import load_ofac_records
        return load_ofac_records(str(self._SDN))

    def test_total_records_nonzero(self) -> None:
        records = self._records()
        assert len(records) > 0, "Parser returned 0 records — namespace or element path is wrong"

    def test_individual_records_nonzero(self) -> None:
        records = self._records()
        individuals = [r for r in records if r.entry_type == "Individual"]
        assert len(individuals) > 0, "Parser found no Individual records"

    def test_some_records_have_aliases(self) -> None:
        records = self._records()
        with_akas = [r for r in records if r.aka_names]
        assert len(with_akas) > 0, "Parser found no records with AKA aliases"


# ── Integration tests (require real OFAC XML) ─────────────────────────────────

@pytest.mark.integration
class TestStep1Integration:
    _SDN = Path("data/raw/ofac/sdn_advanced.xml")
    _CONS = Path("data/raw/ofac/cons_advanced.xml")
    _ACCOUNTS = Path("data/processed/accounts.parquet")

    @pytest.fixture(autouse=True)
    def require_ofac(self) -> None:
        if not self._SDN.exists():
            pytest.skip(
                f"OFAC SDN file not found at {self._SDN}. "
                "Download from https://ofac.treasury.gov/sanctions-list-service"
            )
        if not self._ACCOUNTS.exists():
            pytest.skip(f"accounts.parquet not found at {self._ACCOUNTS}")

    def test_ofac_loads_individuals(self) -> None:
        from aml_copilot.step1_identity.ofac_reader import load_ofac_records
        cons = str(self._CONS) if self._CONS.exists() else None
        records = load_ofac_records(str(self._SDN), cons)
        individuals = [r for r in records if r.entry_type == "Individual"]
        assert len(individuals) >= 1000, f"Expected 1000+ individuals; got {len(individuals)}"

    def test_full_overlay_row_count(self) -> None:
        from aml_copilot.step1_identity.ofac_reader import load_ofac_records
        cons = str(self._CONS) if self._CONS.exists() else None
        records = load_ofac_records(str(self._SDN), cons)
        overlay, _ = build_identity_overlay(str(self._ACCOUNTS), records, seed=42)
        assert len(overlay) == 515_080

    def test_full_no_ofac_name_in_overlay(self) -> None:
        from aml_copilot.step1_identity.ofac_reader import load_ofac_records
        cons = str(self._CONS) if self._CONS.exists() else None
        records = load_ofac_records(str(self._SDN), cons)
        overlay, _ = build_identity_overlay(str(self._ACCOUNTS), records, seed=42)
        raw_names = build_raw_name_set(records)
        overlay_names = set(overlay["name"].to_list())
        assert (overlay_names & raw_names) == set()

    def test_full_hn_max_score_below_threshold(self) -> None:
        from aml_copilot.step1_identity.ofac_reader import load_ofac_records
        from aml_copilot.step1_identity.overlay import _max_score_against_records
        cons = str(self._CONS) if self._CONS.exists() else None
        records = load_ofac_records(str(self._SDN), cons)
        _, rows = build_identity_overlay(str(self._ACCOUNTS), records, seed=42)
        for row in rows:
            if row["gold_is_match"]:
                continue
            max_s, _ = _max_score_against_records(row["assigned_name"], records)
            assert max_s < 0.90, (
                f"HN '{row['assigned_name']}' scores {max_s:.4f} against full OFAC index"
            )
