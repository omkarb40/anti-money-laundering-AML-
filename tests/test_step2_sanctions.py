"""
Unit and integration tests for Step 2: OFAC sanctions screening.

Unit tests (TestOFACIndex, TestComputeScore, TestScreenAccount) use synthetic
OFACRecord objects — no disk I/O required.

Integration tests (TestGroundTruthOracle) require:
  data/raw/ofac/sdn_advanced.xml
  data/fixtures/ground_truth_matches.csv
and are auto-skipped when those files are absent.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from aml_copilot.step1_identity.ofac_reader import OFACRecord
from aml_copilot.step2_sanctions.index import OFACIndex, OFACEntry, build_ofac_index
from aml_copilot.step2_sanctions.screen import (
    MATCH_THRESHOLD,
    ESCALATION_THRESHOLD,
    compute_score,
    screen_account,
)
from aml_copilot.utils.normalize import normalize_name, score_names


# ── Shared synthetic index ────────────────────────────────────────────────────

@pytest.fixture
def sample_records() -> list[OFACRecord]:
    return [
        OFACRecord(
            uid="42", canonical_name="Hassan Ibrahim",
            entry_type="Individual", list_name="SDN",
            aka_names=["Ibrahim Hassan", "H. Ibrahim"],
        ),
        OFACRecord(
            uid="77", canonical_name="García López",
            entry_type="Individual", list_name="SDN",
            aka_names=[],
        ),
        OFACRecord(
            uid="88", canonical_name="John William Smith",
            entry_type="Individual", list_name="SDN",
            aka_names=[],
        ),
        OFACRecord(
            uid="99", canonical_name="Mohammed Hassan",
            entry_type="Individual", list_name="SDN",
            aka_names=["Mohamed Hassan"],
        ),
        OFACRecord(
            uid="55", canonical_name="ACME Corporation",
            entry_type="Entity", list_name="Consolidated",
            aka_names=["Acme Corp", "ACME International"],
        ),
    ]


@pytest.fixture
def sample_index(sample_records) -> OFACIndex:
    return build_ofac_index(sample_records)


# ── TestOFACIndex ─────────────────────────────────────────────────────────────

class TestOFACIndex:
    def test_canonical_in_exact_map(self, sample_index: OFACIndex) -> None:
        assert normalize_name("Hassan Ibrahim") in sample_index.exact_map

    def test_aka_in_exact_map(self, sample_index: OFACIndex) -> None:
        assert normalize_name("Ibrahim Hassan") in sample_index.exact_map

    def test_entry_count_canonical_plus_akas(self, sample_index: OFACIndex) -> None:
        # 5 canonicals + (2+0+0+1+2) AKAs = 10 entries
        assert len(sample_index.all_entries) == 10

    def test_dedup_same_uid_same_norm(self) -> None:
        records = [
            OFACRecord(uid="X", canonical_name="John Smith",
                       entry_type="Individual", list_name="SDN",
                       aka_names=["John Smith"]),  # AKA identical to canonical
        ]
        idx = build_ofac_index(records)
        assert len(idx.all_entries) == 1  # deduped

    def test_is_canonical_true_for_primary_name(self, sample_index: OFACIndex) -> None:
        canonical_entries = [
            e for e in sample_index.all_entries
            if e.uid == "42" and normalize_name(e.raw_name) == normalize_name("Hassan Ibrahim")
        ]
        assert len(canonical_entries) == 1
        assert canonical_entries[0].is_canonical is True

    def test_is_canonical_false_for_aka(self, sample_index: OFACIndex) -> None:
        aka_entries = [
            e for e in sample_index.all_entries
            if e.uid == "42" and normalize_name(e.raw_name) == normalize_name("Ibrahim Hassan")
        ]
        assert len(aka_entries) == 1
        assert aka_entries[0].is_canonical is False

    def test_entity_type_preserved(self, sample_index: OFACIndex) -> None:
        acme = next(e for e in sample_index.all_entries if e.uid == "55" and e.is_canonical)
        assert acme.entry_type == "Entity"
        assert acme.list_name == "Consolidated"

    def test_empty_records_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_ofac_index([])

    def test_two_uids_same_normalized_name(self) -> None:
        records = [
            OFACRecord(uid="A", canonical_name="John Smith",
                       entry_type="Individual", list_name="SDN", aka_names=[]),
            OFACRecord(uid="B", canonical_name="John Smith",
                       entry_type="Entity", list_name="Consolidated", aka_names=[]),
        ]
        idx = build_ofac_index(records)
        hits = idx.exact_map[normalize_name("John Smith")]
        assert {e.uid for e in hits} == {"A", "B"}


# ── TestComputeScore ──────────────────────────────────────────────────────────

class TestComputeScore:
    def test_identical_strings_score_1(self) -> None:
        score, scorer = compute_score("john smith", "john smith")
        assert score == pytest.approx(1.0)

    def test_reversed_tokens_tsr_wins(self) -> None:
        score, scorer = compute_score("smith john william", "john william smith")
        assert score == pytest.approx(1.0)
        assert scorer == "token_sort_ratio"

    def test_close_names_jw_wins(self) -> None:
        # "hassan ibrahm" vs "hassan ibrahim" — one char drop, JW ≥ TSR
        score, scorer = compute_score("hassan ibrahm", "hassan ibrahim")
        assert score >= 0.92
        assert scorer == "jaro_winkler"

    def test_score_is_max_of_jw_and_tsr(self) -> None:
        q, c = "carlos fernandez eduardo", "carlos eduardo fernandez"
        from rapidfuzz.distance import JaroWinkler
        from rapidfuzz import fuzz
        expected = max(
            JaroWinkler.similarity(q, c),
            fuzz.token_sort_ratio(q, c) / 100.0,
        )
        score, _ = compute_score(q, c)
        assert score == pytest.approx(expected)

    def test_completely_different_score_low(self) -> None:
        score, _ = compute_score("xyzzy foobar", "hassan ibrahim")
        assert score < 0.60

    def test_empty_strings_handled(self) -> None:
        score, scorer = compute_score("", "")
        assert 0.0 <= score <= 1.0


# ── TestScreenAccount ─────────────────────────────────────────────────────────

class TestScreenAccount:
    def test_exact_match_score_1(self, sample_index: OFACIndex) -> None:
        hits = screen_account("ACC001", "Hassan Ibrahim", sample_index)
        match = next((h for h in hits if h.ofac_uid == "42"), None)
        assert match is not None
        assert match.match_score == pytest.approx(1.0)
        assert match.scorer_used == "exact"
        assert match.matched_name_type == "canonical"

    def test_alias_searchable_returns_same_uid(self, sample_index: OFACIndex) -> None:
        hits = screen_account("ACC001", "Ibrahim Hassan", sample_index)
        match = next((h for h in hits if h.ofac_uid == "42"), None)
        assert match is not None
        assert match.match_score == pytest.approx(1.0)
        assert match.matched_name_type == "alias"

    def test_aka_in_consolidated_list(self, sample_index: OFACIndex) -> None:
        hits = screen_account("ACC001", "Acme Corp", sample_index)
        match = next((h for h in hits if h.ofac_uid == "55"), None)
        assert match is not None
        assert match.list_source == "Consolidated"

    def test_dedup_by_uid_keeps_highest_score(self, sample_index: OFACIndex) -> None:
        # "ACME Corporation" exact-matches canonical (1.0) and also fuzzy-matches
        # "Acme Corp" / "ACME International" — should produce exactly one hit for uid=55
        hits = screen_account("ACC001", "ACME Corporation", sample_index)
        uid55 = [h for h in hits if h.ofac_uid == "55"]
        assert len(uid55) == 1
        assert uid55[0].match_score == pytest.approx(1.0)

    def test_two_distinct_uids_both_returned(self, sample_index: OFACIndex) -> None:
        # Build a two-uid index where both score ≥ 0.85
        records = [
            OFACRecord(uid="A", canonical_name="Hassan Ibrahim",
                       entry_type="Individual", list_name="SDN", aka_names=[]),
            OFACRecord(uid="B", canonical_name="Hassan Ibrahm",  # one char typo
                       entry_type="Individual", list_name="SDN", aka_names=[]),
        ]
        idx = build_ofac_index(records)
        assert score_names("Hassan Ibrahim", "Hassan Ibrahm") >= 0.85
        hits = screen_account("ACC001", "Hassan Ibrahim", idx)
        uids = {h.ofac_uid for h in hits}
        assert "A" in uids
        assert "B" in uids

    def test_below_threshold_excluded(self, sample_index: OFACIndex) -> None:
        hits = screen_account("ACC001", "Xyzzy Quux Barquux", sample_index)
        assert hits == []

    def test_above_threshold_included(self, sample_index: OFACIndex) -> None:
        # García López normalizes to "garcia lopez" — exact match in index
        hits = screen_account("ACC001", "García López", sample_index)
        assert any(h.ofac_uid == "77" and h.match_score >= MATCH_THRESHOLD for h in hits)

    def test_threshold_constant_is_085(self) -> None:
        assert MATCH_THRESHOLD == pytest.approx(0.85)

    def test_escalation_threshold_constant_is_090(self) -> None:
        assert ESCALATION_THRESHOLD == pytest.approx(0.90)

    def test_empty_name_returns_empty_list(self, sample_index: OFACIndex) -> None:
        assert screen_account("ACC001", "", sample_index) == []

    def test_whitespace_only_name_returns_empty_list(self, sample_index: OFACIndex) -> None:
        assert screen_account("ACC001", "   ", sample_index) == []

    def test_transliteration_match_above_085(self, sample_index: OFACIndex) -> None:
        # "Mohamed Hassan" is an AKA of uid=99 "Mohammed Hassan"
        hits = screen_account("ACC001", "Mohamed Hassan", sample_index)
        match = next((h for h in hits if h.ofac_uid == "99"), None)
        assert match is not None
        assert match.match_score >= 0.85

    def test_token_sort_ratio_handles_reorder(self, sample_index: OFACIndex) -> None:
        # "Smith William John" is "John William Smith" reordered — TSR = 1.0
        hits = screen_account("ACC001", "Smith William John", sample_index)
        match = next((h for h in hits if h.ofac_uid == "88"), None)
        assert match is not None
        assert match.match_score >= 0.90
        assert match.scorer_used == "token_sort_ratio"

    def test_results_sorted_descending_by_score(self, sample_index: OFACIndex) -> None:
        hits = screen_account("ACC001", "Hassan Ibrahim", sample_index)
        scores = [h.match_score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_assigned_name_not_in_info_logs(
        self, sample_index: OFACIndex, caplog: pytest.LogCaptureFixture
    ) -> None:
        name = "Hassan Ibrahim"
        with caplog.at_level(logging.INFO):
            screen_account("ACC001", name, sample_index)
        for record in caplog.records:
            assert name not in record.message, (
                f"assigned_name '{name}' leaked into INFO log: '{record.message}'"
            )

    def test_account_id_in_every_hit(self, sample_index: OFACIndex) -> None:
        hits = screen_account("ACC999", "Hassan Ibrahim", sample_index)
        assert all(h.account_id == "ACC999" for h in hits)

    def test_match_score_between_0_and_1(self, sample_index: OFACIndex) -> None:
        hits = screen_account("ACC001", "Hassan Ibrahim", sample_index)
        assert all(0.0 <= h.match_score <= 1.0 for h in hits)

    def test_typo_ocr_match_above_085(self) -> None:
        records = [
            OFACRecord(uid="Z", canonical_name="Roberto Garcia",
                       entry_type="Individual", list_name="SDN", aka_names=[]),
        ]
        idx = build_ofac_index(records)
        # "R0berto Garcia" — OCR '0' for 'o'
        hits = screen_account("ACC001", "R0berto Garcia", idx)
        match = next((h for h in hits if h.ofac_uid == "Z"), None)
        assert match is not None
        assert match.match_score >= 0.85


# ── TestGroundTruthOracle (integration) ───────────────────────────────────────

@pytest.mark.integration
class TestGroundTruthOracle:
    _SDN = Path("data/raw/ofac/sdn_advanced.xml")
    _CONS = Path("data/raw/ofac/cons_advanced.xml")
    _GT = Path("data/fixtures/ground_truth_matches.csv")

    @pytest.fixture(autouse=True)
    def require_data(self) -> None:
        if not self._SDN.exists():
            pytest.skip(f"OFAC SDN not found: {self._SDN}")
        if not self._GT.exists():
            pytest.skip(f"Ground truth CSV not found: {self._GT}")

    @pytest.fixture
    def real_index(self) -> OFACIndex:
        from aml_copilot.step1_identity.ofac_reader import load_ofac_records
        cons = str(self._CONS) if self._CONS.exists() else None
        records = load_ofac_records(str(self._SDN), cons)
        return build_ofac_index(records)

    def test_index_entry_count_above_10k(self, real_index: OFACIndex) -> None:
        assert len(real_index.all_entries) > 10_000, (
            f"Index too small: {len(real_index.all_entries)} entries — "
            "OFAC parser may have failed"
        )

    def test_alias_count_nonzero(self, real_index: OFACIndex) -> None:
        alias_entries = [e for e in real_index.all_entries if not e.is_canonical]
        assert len(alias_entries) > 0

    def test_all_20_positives_match_ofac_uid(self, real_index: OFACIndex) -> None:
        from aml_copilot.step1_identity.ground_truth import load_ground_truth
        rows = load_ground_truth(self._GT)
        failures: list[str] = []
        for row in rows:
            if not row.gold_is_match:
                continue
            hits = screen_account("TEST", row.assigned_name, real_index)
            target = next((h for h in hits if h.ofac_uid == row.ofac_uid), None)
            if target is None:
                top = [(h.ofac_uid, round(h.match_score, 4)) for h in hits[:3]]
                failures.append(
                    f"[{row.match_flavor}] uid={row.ofac_uid}: no hit "
                    f"(top3={top})"
                )
            elif target.match_score < row.expected_score_min:
                failures.append(
                    f"[{row.match_flavor}] uid={row.ofac_uid}: "
                    f"score={target.match_score:.4f} < min={row.expected_score_min}"
                )
        assert not failures, "TP failures:\n" + "\n".join(failures)

    def test_all_30_hard_negatives_below_escalation(self, real_index: OFACIndex) -> None:
        from aml_copilot.step1_identity.ground_truth import load_ground_truth
        rows = load_ground_truth(self._GT)
        failures: list[str] = []
        max_score = 0.0
        for row in rows:
            if row.gold_is_match:
                continue
            hits = screen_account("TEST", row.assigned_name, real_index)
            target = next((h for h in hits if h.ofac_uid == row.ofac_uid), None)
            if target is not None:
                max_score = max(max_score, target.match_score)
                if target.match_score >= ESCALATION_THRESHOLD:
                    failures.append(
                        f"HN uid={row.ofac_uid} name='{row.assigned_name}': "
                        f"score={target.match_score:.4f} >= {ESCALATION_THRESHOLD} "
                        f"(would false-escalate)"
                    )
        assert not failures, "HN escalation failures:\n" + "\n".join(failures)
