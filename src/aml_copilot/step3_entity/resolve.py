"""
Step 3: Entity resolution.

Builds an in-memory undirected transaction graph from the HI-Small transaction
DataFrame, an account profile lookup from identity_overlay.parquet, and a
typology label map from HI-Small_Patterns.txt.

resolve_entity() returns an EntityChain for a single account_id:
  - name, country, kyc_risk from the overlay
  - hop-1 direct counterparties from the graph
  - hop-2 second-degree counterparties, capped at HOP2_CAP
  - pattern_label from the typology map if the account is in the patterns file

The 'name' field comes from identity_overlay.parquet, which is guaranteed
OFAC-safe by Step 1's safety check. No OFAC raw names appear here.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import polars as pl

from aml_copilot.schemas import EntityChain

logger = logging.getLogger(__name__)

HOP2_CAP: int = 50


# ── Graph construction ────────────────────────────────────────────────────────

def build_transaction_graph(df: pl.DataFrame) -> dict[str, set[str]]:
    """
    Build an undirected adjacency dict from a transaction DataFrame.

    Uses Polars group_by to avoid Python-level row iteration over 5M+ rows.
    Self-loops (from_account == to_account) are excluded.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain 'from_account' and 'to_account' string columns.

    Returns
    -------
    dict[str, set[str]]
        {account_id: {counterparty_ids}} — both directions per edge,
        deduplicated.
    """
    non_self = df.filter(pl.col("from_account") != pl.col("to_account"))

    # Emit (account, counterparty) for both directions, then group
    pairs = pl.concat([
        non_self.select(
            pl.col("from_account").alias("account"),
            pl.col("to_account").alias("counterparty"),
        ),
        non_self.select(
            pl.col("to_account").alias("account"),
            pl.col("from_account").alias("counterparty"),
        ),
    ])

    grouped = (
        pairs
        .group_by("account")
        .agg(pl.col("counterparty").unique())
    )

    adjacency: dict[str, set[str]] = {}
    for row in grouped.iter_rows(named=True):
        adjacency[row["account"]] = set(row["counterparty"])

    return adjacency


# ── Overlay map ───────────────────────────────────────────────────────────────

def build_overlay_map(overlay_parquet: str | Path) -> dict[str, dict]:
    """
    Load identity_overlay.parquet into a plain Python dict for O(1) lookup.

    Returns
    -------
    dict[str, dict]
        {account_id: {"name": str, "country": str|None, "kyc_risk": str|None}}
    """
    df = pl.read_parquet(overlay_parquet)
    result: dict[str, dict] = {}
    for row in df.iter_rows(named=True):
        result[row["account_id"]] = {
            "name": row["name"],
            "country": row.get("country"),
            "kyc_risk": row.get("kyc_risk"),
        }
    return result


# ── Pattern map ───────────────────────────────────────────────────────────────

_BEGIN_RE = re.compile(
    r"BEGIN\s+LAUNDERING\s+ATTEMPT\s*-\s*([^:]+)", re.IGNORECASE
)
_END_RE = re.compile(r"END\s+LAUNDERING\s+ATTEMPT", re.IGNORECASE)


def parse_patterns_file(path: str | Path) -> dict[str, list[str]]:
    """
    Parse HI-Small_Patterns.txt into {typology_name: [account_id, ...]}.

    The file uses BEGIN/END LAUNDERING ATTEMPT - TYPE_NAME: ... blocks.
    Each line within a block is a transaction CSV row; account IDs are at
    column indices 2 (from_account) and 4 (to_account).

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Patterns file not found: {path}")

    result: dict[str, list[str]] = {}
    current_typology: Optional[str] = None
    current_accounts: set[str] = set()

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            m = _BEGIN_RE.match(line)
            if m:
                # e.g. "FAN-OUT:  Max 16-degree..." → strip colon and trailing text
                raw = m.group(1).strip()
                current_typology = raw.split(":")[0].strip()
                current_accounts = set()
                continue

            if _END_RE.match(line):
                if current_typology is not None:
                    existing = set(result.get(current_typology, []))
                    new_ids = sorted(current_accounts - existing)
                    result.setdefault(current_typology, []).extend(new_ids)
                current_typology = None
                continue

            if current_typology is not None and "," in line:
                parts = line.split(",")
                if len(parts) >= 5:
                    from_acct = parts[2].strip()
                    to_acct = parts[4].strip()
                    if from_acct:
                        current_accounts.add(from_acct)
                    if to_acct:
                        current_accounts.add(to_acct)

    return result


def build_pattern_map(patterns: dict[str, list[str]]) -> dict[str, str]:
    """
    Invert {typology: [account_ids]} → {account_id: typology_name}.

    If an account appears in multiple typologies, the alphabetically first
    typology name is used (deterministic).

    Logs a WARNING if the resulting map is empty.
    """
    result: dict[str, str] = {}
    for typology in sorted(patterns.keys()):
        for account_id in patterns[typology]:
            if account_id not in result:
                result[account_id] = typology

    if not result:
        logger.warning(
            "Pattern map is empty — check that patterns file was parsed correctly"
        )

    return result


# ── Entity resolution ─────────────────────────────────────────────────────────

def resolve_entity(
    account_id: str,
    adjacency: dict[str, set[str]],
    overlay_map: dict[str, dict],
    pattern_map: dict[str, str],
) -> EntityChain:
    """
    Return an EntityChain for account_id.

    Parameters
    ----------
    account_id : str
        The account to resolve.
    adjacency : dict[str, set[str]]
        Undirected transaction graph from build_transaction_graph().
    overlay_map : dict[str, dict]
        Account profile lookup from build_overlay_map().
    pattern_map : dict[str, str]
        Typology label lookup from build_pattern_map().

    Returns
    -------
    EntityChain
        hop2_counterparties capped at HOP2_CAP (sorted alphabetically).
        If account_id is not in overlay_map, name="UNKNOWN" and KYC fields
        are None; a WARNING is logged.
    """
    # ── Profile lookup ────────────────────────────────────────────────────────
    profile = overlay_map.get(account_id)
    if profile is None:
        logger.warning(
            "Account %s not in overlay — returning UNKNOWN profile", account_id
        )
        name: str = "UNKNOWN"
        country: Optional[str] = None
        kyc_risk: Optional[str] = None
    else:
        name = profile["name"]
        country = profile.get("country")
        kyc_risk = profile.get("kyc_risk")

    # ── Hop 1: direct counterparties ─────────────────────────────────────────
    hop1_set: set[str] = adjacency.get(account_id, set())
    hop1: list[str] = sorted(hop1_set)

    # ── Hop 2: BFS level-2 with visited set ──────────────────────────────────
    # visited prevents: self appearing in hop-2, hop-1 accounts appearing in
    # hop-2, and infinite loops on cycles.
    visited: set[str] = {account_id} | hop1_set
    hop2_set: set[str] = set()
    for cp in hop1_set:
        for cp2 in adjacency.get(cp, set()):
            if cp2 not in visited:
                hop2_set.add(cp2)

    hop2: list[str] = sorted(hop2_set)[:HOP2_CAP]

    # ── Pattern label ─────────────────────────────────────────────────────────
    pattern_label: Optional[str] = pattern_map.get(account_id)

    return EntityChain(
        account_id=account_id,
        name=name,
        country=country,
        kyc_risk=kyc_risk,
        hop1_counterparties=hop1,
        hop2_counterparties=hop2,
        pattern_label=pattern_label,
    )
