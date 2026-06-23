"""
Minimal OFAC Advanced XML reader for Step 1 identity overlay construction.

Supports OFAC Advanced XML v3 schema:
  namespace: https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/ADVANCED_XML
  record element: DistinctParties/DistinctParty

Produces OFACRecord objects (one per SDN/Consolidated entity) with the full
name list (canonical + AKAs). This is intentionally separate from
step2_sanctions.index.OFACEntry, which is flat (one entry per normalized name).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

_LATIN_SCRIPT_ID = "215"


@dataclass
class OFACRecord:
    uid: str
    canonical_name: str   # raw, pre-normalization — never logged above DEBUG
    aka_names: list[str]  # raw AKA names
    entry_type: str       # "Individual" | "Entity"
    list_name: str        # "SDN" | "Consolidated"

    @property
    def all_raw_names(self) -> list[str]:
        return [self.canonical_name] + self.aka_names


def _extract_latin_name(alias_el: ET.Element, ns: str) -> str | None:
    """
    Collect all Latin-script NamePartValues from a single Alias element and
    join them into one name string.  Returns None if no Latin text found.
    """
    parts: list[str] = []
    for doc_name in alias_el.findall(f"{{{ns}}}DocumentedName"):
        for dnp in doc_name.findall(f"{{{ns}}}DocumentedNamePart"):
            for npv in dnp.findall(f"{{{ns}}}NamePartValue"):
                if npv.attrib.get("ScriptID") == _LATIN_SCRIPT_ID and npv.text:
                    parts.append(npv.text.strip())
    return " ".join(parts) if parts else None


def _parse_xml(path: Path, list_name: str) -> list[OFACRecord]:
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse {path}: {exc}") from exc

    root = tree.getroot()
    ns = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""

    # Build PartySubTypeID → PartyTypeID mapping from the XML's own reference set.
    # PartyTypeID "1" = Individual; all others are Entity / Vessel / Aircraft.
    subtype_to_partytype: dict[str, str] = {}
    ref_sets = root.find(f"{{{ns}}}ReferenceValueSets")
    if ref_sets is not None:
        pstv = ref_sets.find(f"{{{ns}}}PartySubTypeValues")
        if pstv is not None:
            for item in pstv:
                sid = item.attrib.get("ID", "")
                ptid = item.attrib.get("PartyTypeID", "")
                if sid and ptid:
                    subtype_to_partytype[sid] = ptid

    dp_container = root.find(f"{{{ns}}}DistinctParties")
    if dp_container is None:
        return []

    records: list[OFACRecord] = []
    for party in dp_container.findall(f"{{{ns}}}DistinctParty"):
        uid = party.attrib.get("FixedRef", "")
        if not uid:
            continue

        profile = party.find(f"{{{ns}}}Profile")
        if profile is None:
            continue

        subtype_id = profile.attrib.get("PartySubTypeID", "")
        party_type_id = subtype_to_partytype.get(subtype_id, "")
        entry_type = "Individual" if party_type_id == "1" else "Entity"

        identity = profile.find(f"{{{ns}}}Identity")
        if identity is None:
            continue

        canonical_name: str | None = None
        aka_names: list[str] = []

        for alias in identity.findall(f"{{{ns}}}Alias"):
            is_primary = alias.attrib.get("Primary", "false").lower() == "true"
            name = _extract_latin_name(alias, ns)
            if not name:
                continue
            if is_primary:
                canonical_name = name
            else:
                aka_names.append(name)

        if not canonical_name:
            continue

        records.append(OFACRecord(
            uid=uid,
            canonical_name=canonical_name,
            aka_names=aka_names,
            entry_type=entry_type,
            list_name=list_name,
        ))

    return records


def load_ofac_records(
    sdn_path: str | Path,
    cons_path: str | Path | None = None,
) -> list[OFACRecord]:
    """
    Parse sdn_advanced.xml and optionally cons_advanced.xml.
    Returns combined list of OFACRecord; individuals first.

    Raises FileNotFoundError if sdn_path is absent.
    """
    sdn_path = Path(sdn_path)
    if not sdn_path.exists():
        raise FileNotFoundError(
            f"OFAC SDN file not found: {sdn_path}\n"
            "Download sdn_advanced.xml from https://ofac.treasury.gov/sanctions-list-service "
            "and place it under data/raw/ofac/"
        )

    records = _parse_xml(sdn_path, "SDN")

    if cons_path is not None:
        cons_path = Path(cons_path)
        if cons_path.exists():
            records.extend(_parse_xml(cons_path, "Consolidated"))

    return records


def build_raw_name_set(records: list[OFACRecord]) -> set[str]:
    """All raw canonical + AKA names across all records. Used for collision detection."""
    names: set[str] = set()
    for r in records:
        names.add(r.canonical_name)
        names.update(r.aka_names)
    return names
