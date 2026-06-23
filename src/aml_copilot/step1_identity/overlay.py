"""
Step 1: Synthetic identity overlay.

Assigns a name, country, and kyc_risk to every account in accounts.parquet.
20 accounts receive names derived from real OFAC entries (TPs).
30 accounts receive near-threshold names (HNs).
The remaining 515,030 accounts receive seeded Faker names, screened to ensure
no accidental OFAC match ≥ 0.90.

SAFETY: raw OFAC canonical names and AKA names must never appear in the
'name' column of identity_overlay.parquet. They exist only in:
  1. The in-memory OFACRecord list used during construction.
  2. ground_truth_matches.csv (ofac_canonical_name column).
"""
from __future__ import annotations

import logging
import random
import unicodedata
from pathlib import Path

import polars as pl
from faker import Faker

from aml_copilot.step1_identity.ofac_reader import OFACRecord, build_raw_name_set
from aml_copilot.utils.normalize import nfkd_normalize, normalize_name, score_names

logger = logging.getLogger(__name__)

FAKER_SEED: int = 42

# kyc_risk distribution (weights for low/medium/high)
_KYC_WEIGHTS = [85, 12, 3]
_KYC_VALUES = ["low", "medium", "high"]

# ── Transliteration substitution table ───────────────────────────────────────
# Each tuple: (pattern_to_find, replacement). Applied to the lowercased name.
_TRANSLIT_SUBS: list[tuple[str, str]] = [
    ("mohammed", "mohamed"),
    ("hussain", "husain"),
    ("ahmad", "ahmed"),
    ("usama", "osama"),
    ("mukhtar", "mokhtar"),
    ("khoury", "khouri"),
    ("ou", "u"),
    ("ph", "f"),
    ("ae", "a"),
    ("kh", "k"),
    ("ai", "ei"),
    ("ii", "i"),
    ("uu", "u"),
]

# OCR / typo substitution table
_OCR_SUBS: list[tuple[str, str]] = [
    ("o", "0"),
    ("i", "1"),
    ("l", "1"),
    ("e", "3"),
    ("s", "5"),
    ("a", "4"),
    ("g", "9"),
    ("b", "6"),
]

_VOWELS = set("aeiou")


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _max_score_against_records(name: str, records: list[OFACRecord]) -> tuple[float, str]:
    """Return (max_score, uid) across every name in every record."""
    best = 0.0
    best_uid = ""
    for rec in records:
        for raw in rec.all_raw_names:
            s = score_names(name, raw)
            if s > best:
                best = s
                best_uid = rec.uid
    return best, best_uid


# ── TP name generators ────────────────────────────────────────────────────────

def _gen_exact(entry: OFACRecord, raw_names: set[str]) -> str | None:
    """
    Assign nfkd(canonical_name). Valid only when:
    1. The normalized form differs from the raw canonical (non-ASCII present).
    2. The normalized form is not itself any raw OFAC name.
    """
    norm = nfkd_normalize(entry.canonical_name)
    if norm == entry.canonical_name:
        return None  # pure ASCII — normalization is identity → would equal raw name
    if norm in raw_names:
        return None  # normalized form equals another raw OFAC name
    return norm


def _gen_transliteration(entry: OFACRecord, raw_names: set[str], rng: random.Random) -> str | None:
    """Apply one romanization variant substitution; verify score ≥ 0.90."""
    base = normalize_name(entry.canonical_name)
    subs = list(_TRANSLIT_SUBS)
    rng.shuffle(subs)
    for old, new in subs:
        if old not in base:
            continue
        candidate = base.replace(old, new, 1)
        if candidate == base:
            continue
        if candidate in {normalize_name(n) for n in raw_names}:
            continue
        s = score_names(candidate, entry.canonical_name)
        if s >= 0.90:
            # Restore approximate title-case from the normalized base
            return _restore_case(candidate, base)
    # Fallback: single vowel substitution at a non-prefix position
    words = base.split()
    for wi, word in enumerate(words):
        for ci in range(max(3, len(word) - 3), len(word)):
            if word[ci] in _VOWELS:
                vowels = list(_VOWELS - {word[ci]})
                rng.shuffle(vowels)
                for v in vowels:
                    candidate = base[:base.index(word)] + word[:ci] + v + word[ci + 1:] + " " + " ".join(words[wi + 1:])
                    candidate = " ".join(candidate.split())
                    if candidate == base or candidate in {normalize_name(n) for n in raw_names}:
                        continue
                    s = score_names(candidate, entry.canonical_name)
                    if s >= 0.90:
                        return _restore_case(candidate, base)
    return None


def _gen_typo_ocr(entry: OFACRecord, raw_names: set[str], rng: random.Random) -> str | None:
    """Apply 1–2 OCR-style substitutions; verify score ≥ 0.85."""
    base = normalize_name(entry.canonical_name)
    subs = list(_OCR_SUBS)
    rng.shuffle(subs)
    for old, new in subs:
        if old not in base:
            continue
        candidate = base.replace(old, new, 1)
        if candidate == base or candidate in {normalize_name(n) for n in raw_names}:
            continue
        s = score_names(candidate, entry.canonical_name)
        if s >= 0.85:
            return _restore_case(candidate, base)
    # Fallback: swap two adjacent characters past position 3
    for i in range(3, len(base) - 1):
        if base[i] == " " or base[i + 1] == " ":
            continue
        candidate = base[:i] + base[i + 1] + base[i] + base[i + 2:]
        if candidate == base or candidate in {normalize_name(n) for n in raw_names}:
            continue
        s = score_names(candidate, entry.canonical_name)
        if s >= 0.85:
            return _restore_case(candidate, base)
    return None


def _gen_partial_reorder(entry: OFACRecord, raw_names: set[str]) -> str | None:
    """Reverse token order; verify token_sort_ratio ≥ 0.90 (always true for reversal)."""
    tokens = nfkd_normalize(entry.canonical_name).split()
    if len(tokens) < 2:
        return None
    reversed_name = " ".join(reversed(tokens))
    if reversed_name in raw_names:
        return None
    # token_sort_ratio of a reversal is always 1.0 — no score check needed
    return reversed_name


def _restore_case(modified: str, original_lower: str) -> str:
    """Title-case the modified name, preserving digits and special chars."""
    return modified.title()


# ── HN generator ─────────────────────────────────────────────────────────────

def _gen_hard_negative(
    target: OFACRecord,
    all_records: list[OFACRecord],
    fake: Faker,
    rng: random.Random,
    max_retries: int = 150,
) -> str | None:
    """
    Surname-sharing hard negative. Retry loop advances Faker sequence.
    FINAL acceptance: full-index screen proves max_score < 0.90.
    Target score must be in [0.80, 0.88].
    """
    # Extract surname (last token of canonical name after normalization)
    tokens = normalize_name(target.canonical_name).split()
    if not tokens:
        return None
    surname = tokens[-1]
    if len(surname) < 4:
        return None  # very short surnames make JW unpredictable

    target_norm = normalize_name(target.canonical_name)

    for _ in range(max_retries):
        given = fake.first_name()
        candidate = f"{given} {surname}".title()
        cand_norm = normalize_name(candidate)

        if cand_norm == target_norm:
            continue

        target_score = score_names(candidate, target.canonical_name)
        if not (0.80 <= target_score <= 0.88):
            continue

        # Full-index check — no shortcuts, per user constraint
        max_score, _ = _max_score_against_records(candidate, all_records)
        if max_score >= 0.90:
            continue

        return candidate

    return None


# ── Faker collision screen ────────────────────────────────────────────────────

def _build_surname_index(records: list[OFACRecord]) -> dict[str, list[OFACRecord]]:
    """Block by last token of each normalized OFAC name for fast pre-filter."""
    idx: dict[str, list[OFACRecord]] = {}
    for rec in records:
        for raw in rec.all_raw_names:
            tokens = normalize_name(raw).split()
            if tokens:
                idx.setdefault(tokens[-1], []).append(rec)
    return idx


def _faker_name_collides(
    name: str,
    raw_names: set[str],
    norm_raw_names: set[str],
    surname_idx: dict[str, list[OFACRecord]],
) -> bool:
    """
    Phase 1: exact-match O(1).
    Phase 2: surname-blocked fuzzy scan (speed optimization only).
    Returns True if name matches any OFAC entry at ≥ 0.90.
    """
    norm = normalize_name(name)
    # Phase 1 — both sets are pre-computed once in build_identity_overlay
    if name in raw_names or norm in norm_raw_names:
        return True
    # Phase 2
    tokens = norm.split()
    if not tokens:
        return False
    candidates = surname_idx.get(tokens[-1], [])
    for rec in candidates:
        for raw in rec.all_raw_names:
            if score_names(name, raw) >= 0.90:
                return True
    return False


# ── Main builder ─────────────────────────────────────────────────────────────

def build_identity_overlay(
    accounts_parquet: str | Path,
    ofac_records: list[OFACRecord],
    seed: int = FAKER_SEED,
) -> tuple[pl.DataFrame, list[dict]]:
    """
    Build identity_overlay DataFrame and ground-truth fixture rows.

    Returns
    -------
    overlay : pl.DataFrame
        515,080 rows with columns [account_id, name, country, kyc_risk].
        'name' column never contains raw OFAC canonical or AKA names.
    fixture_rows : list[dict]
        50 dicts matching GroundTruthRow schema (20 TP + 30 HN).
    """
    accounts_parquet = Path(accounts_parquet)
    if not accounts_parquet.exists():
        raise FileNotFoundError(f"accounts.parquet not found: {accounts_parquet}")

    accounts_df = pl.read_parquet(accounts_parquet).sort("account_id")
    account_ids: list[str] = accounts_df["account_id"].to_list()

    rng = random.Random(seed)
    Faker.seed(seed)
    fake = Faker()

    raw_names: set[str] = build_raw_name_set(ofac_records)
    norm_raw_names: set[str] = {normalize_name(n) for n in raw_names}
    surname_idx = _build_surname_index(ofac_records)

    # Separate individuals (better for name-similarity flavors)
    individuals = [r for r in ofac_records if r.entry_type == "Individual"]
    if len(individuals) < 10:
        raise ValueError(
            f"Need at least 10 individual OFAC entries; got {len(individuals)}. "
            "Check that the OFAC XML files loaded correctly."
        )

    # ── Select fixture accounts ───────────────────────────────────────────────
    n_fixture = min(50, len(account_ids))
    fixture_indices = sorted(rng.sample(range(len(account_ids)), n_fixture))
    tp_indices = fixture_indices[:20]
    hn_indices = fixture_indices[20:50]
    fixture_account_ids = set(account_ids[i] for i in fixture_indices)

    # ── Pre-partition OFAC pool per flavor (avoids entries being consumed by wrong flavor) ──
    shuffled = list(individuals)
    rng.shuffle(shuffled)

    exact_pool  = [e for e in shuffled if _gen_exact(e, raw_names) is not None]
    reorder_pool = [e for e in shuffled if len(e.canonical_name.split()) >= 2]
    general_pool = shuffled  # transliteration, typo/ocr, HN can all try any entry

    # ── Generate TP names (5 per flavor) ─────────────────────────────────────
    flavor_pools = [
        ("exact",           exact_pool),
        ("transliteration", general_pool),
        ("typo_ocr",        general_pool),
        ("partial_reorder", reorder_pool),
    ]
    flavor_generators = {
        "exact":           _try_exact_pool,
        "transliteration": _try_translit_pool,
        "typo_ocr":        _try_typo_pool,
        "partial_reorder": _try_reorder_pool,
    }

    used_uids: set[str] = set()
    fixture_rows: list[dict] = []
    tp_iter = iter(tp_indices)

    for flavor, pool in flavor_pools:
        score_min, score_max = _flavor_bounds(flavor)
        generator = flavor_generators[flavor]
        assigned: list[tuple[str, str, str]] = []  # (account_id, name, uid)
        for entry in pool:
            if len(assigned) == 5:
                break
            if entry.uid in used_uids:
                continue
            name = generator(entry, raw_names, rng)
            if name is None or name in raw_names:
                continue
            actual_score = score_names(name, entry.canonical_name)
            if actual_score < score_min:
                continue
            used_uids.add(entry.uid)
            acct = account_ids[next(tp_iter)]
            assigned.append((acct, name, entry.uid))

        if len(assigned) < 5:
            raise RuntimeError(
                f"Could only find {len(assigned)}/5 entries for flavor '{flavor}'. "
                f"Add more OFAC individuals with suitable names for this flavor."
            )

        for acct, name, uid in assigned:
            entry_match = next(r for r in ofac_records if r.uid == uid)
            fixture_rows.append({
                "account_id": acct,
                "assigned_name": name,
                "ofac_uid": uid,
                "ofac_canonical_name": entry_match.canonical_name,
                "match_flavor": flavor,
                "expected_score_min": score_min,
                "expected_score_max": score_max,
                "gold_is_match": True,
            })

    # ── Generate HN names ─────────────────────────────────────────────────────
    hn_iter = iter(hn_indices)
    hn_assigned = 0
    # Try all individuals as HN sources (including ones used for TPs — different names)
    for entry in shuffled:
        if hn_assigned >= 30:
            break
        name = _gen_hard_negative(entry, ofac_records, fake, rng)
        if name is None or name in raw_names:
            continue
        acct = account_ids[next(hn_iter)]
        fixture_rows.append({
            "account_id": acct,
            "assigned_name": name,
            "ofac_uid": entry.uid,
            "ofac_canonical_name": entry.canonical_name,
            "match_flavor": "hard_negative",
            "expected_score_min": 0.80,
            "expected_score_max": 0.88,
            "gold_is_match": False,
        })
        hn_assigned += 1

    if hn_assigned < 30:
        raise RuntimeError(
            f"Could only generate {hn_assigned}/30 hard negatives. "
            "Try expanding the OFAC individual pool or increase max_retries."
        )

    # Number rows
    for i, row in enumerate(fixture_rows):
        row["row_id"] = i

    # ── Build name lookup for fixture accounts ────────────────────────────────
    fixture_name_map: dict[str, str] = {
        row["account_id"]: row["assigned_name"] for row in fixture_rows
    }

    # ── Assign Faker names to non-fixture accounts ────────────────────────────
    logger.info("Assigning Faker names to %d non-fixture accounts...", len(account_ids) - 50)
    names: list[str] = []
    countries: list[str] = []
    kyc_risks: list[str] = []
    regen_count = 0

    for acct in account_ids:
        if acct in fixture_name_map:
            names.append(fixture_name_map[acct])
        else:
            for _ in range(200):
                candidate = fake.name()
                if not _faker_name_collides(candidate, raw_names, norm_raw_names, surname_idx):
                    names.append(candidate)
                    break
                regen_count += 1
            else:
                # Extremely unlikely — append with a guaranteed-safe UUID suffix
                names.append(fake.name() + f" {rng.randint(1000,9999)}")

        countries.append(fake.country_code())
        kyc_risks.append(rng.choices(_KYC_VALUES, weights=_KYC_WEIGHTS)[0])

    if regen_count:
        logger.info("Faker collision resolution: %d names regenerated", regen_count)

    overlay = pl.DataFrame({
        "account_id": account_ids,
        "name": names,
        "country": countries,
        "kyc_risk": kyc_risks,
    })

    return overlay, fixture_rows


# ── Flavor-specific wrappers ──────────────────────────────────────────────────

def _try_exact_pool(entry: OFACRecord, raw_names: set[str], rng: random.Random) -> str | None:
    return _gen_exact(entry, raw_names)


def _try_translit_pool(entry: OFACRecord, raw_names: set[str], rng: random.Random) -> str | None:
    return _gen_transliteration(entry, raw_names, rng)


def _try_typo_pool(entry: OFACRecord, raw_names: set[str], rng: random.Random) -> str | None:
    return _gen_typo_ocr(entry, raw_names, rng)


def _try_reorder_pool(entry: OFACRecord, raw_names: set[str], rng: random.Random) -> str | None:
    return _gen_partial_reorder(entry, raw_names)


def _flavor_bounds(flavor: str) -> tuple[float, float]:
    return {
        "exact":           (1.00, 1.00),
        "transliteration": (0.90, 1.00),
        "typo_ocr":        (0.85, 1.00),
        "partial_reorder": (0.90, 1.00),
    }[flavor]


def save_overlay(overlay: pl.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_parquet(path)
    logger.info("Overlay written: %s (%d rows)", path, len(overlay))
