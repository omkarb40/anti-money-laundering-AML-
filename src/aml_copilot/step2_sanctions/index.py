"""
Step 2: OFAC screening index.

Builds a flat, normalized, AKA-expanded index from the OFACRecord list
produced by step1_identity.ofac_reader. Each canonical name and each alias
becomes one OFACEntry. The index supports O(1) exact-name lookup and a
linear fuzzy scan.

raw_name is carried in OFACEntry for internal traceability but must never
be logged at INFO level or above by any caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aml_copilot.step1_identity.ofac_reader import OFACRecord
from aml_copilot.utils.normalize import normalize_name


@dataclass
class OFACEntry:
    uid: str
    list_name: str       # "SDN" | "Consolidated"
    entry_type: str      # "Individual" | "Entity"
    raw_name: str        # never logged above DEBUG
    normalized_name: str
    is_canonical: bool   # True = primary name, False = AKA


@dataclass
class OFACIndex:
    # normalized_name → all entries with that normalized form (usually one,
    # but two different UIDs can share the same normalized name)
    exact_map: dict[str, list[OFACEntry]]
    # full flat list for fuzzy scan — (uid, normalized_name) is unique
    all_entries: list[OFACEntry]


def build_ofac_index(records: list[OFACRecord]) -> OFACIndex:
    """
    Expand OFACRecord list into a flat OFACIndex.

    For each record: one entry for the canonical name + one per AKA name.
    Deduplication: (uid, normalized_name) pairs appear at most once in
    all_entries (handles identical AKA spellings for the same entity).

    Raises ValueError if records is empty.
    """
    if not records:
        raise ValueError("OFAC index: records list is empty — check parser output")

    seen: set[tuple[str, str]] = set()
    all_entries: list[OFACEntry] = []
    exact_map: dict[str, list[OFACEntry]] = {}

    for record in records:
        name_variants: list[tuple[str, bool]] = [
            (record.canonical_name, True),
        ] + [(aka, False) for aka in record.aka_names]

        for raw_name, is_canonical in name_variants:
            if not raw_name or not raw_name.strip():
                continue
            norm = normalize_name(raw_name)
            if not norm:
                continue
            key = (record.uid, norm)
            if key in seen:
                continue
            seen.add(key)

            entry = OFACEntry(
                uid=record.uid,
                list_name=record.list_name,
                entry_type=record.entry_type,
                raw_name=raw_name,
                normalized_name=norm,
                is_canonical=is_canonical,
            )
            all_entries.append(entry)
            exact_map.setdefault(norm, []).append(entry)

    return OFACIndex(exact_map=exact_map, all_entries=all_entries)
