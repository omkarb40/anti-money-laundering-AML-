from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OFACEntry:
    uid: str
    canonical_name: str
    normalized_name: str
    list_name: str         # "SDN" | "Consolidated"
    entity_type: str       # "individual" | "entity"


def build_ofac_index(sdn_path: str, cons_path: str) -> dict[str, OFACEntry]:
    """
    Parse both OFAC XML files, expand all AKA/alias fields (each variant becomes
    a separate index entry sharing the same uid), and return a dict keyed by
    normalized_name for O(1) exact-match lookup.
    """
    ...


def parse_sdn_xml(path: str) -> list[OFACEntry]:
    """Parse sdn_advanced.xml; expand <aka> blocks into individual OFACEntry objects."""
    ...


def parse_cons_xml(path: str) -> list[OFACEntry]:
    """Parse cons_advanced.xml; same AKA expansion logic as parse_sdn_xml."""
    ...
