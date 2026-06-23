# Domain constants — frozen before eval set construction.
# SHA-256 of this file is recorded in artifacts/checksums.sha256.
# Do not modify after freeze.
#
# Threshold sources:
#   STRUCT_THRESHOLD : US Bank Secrecy Act, 31 USC §5324 (CTR threshold)
#   PASSTHROUGH_*    : FATF Recommendation 10 (layering / rapid in-out)
#   FAN_*            : IBM AMLSim fan-burst characteristics; avg degree = 3.0
#   HIGH_RISK_COUNTRIES : FATF Public Statement, June 2023

# ── Structuring (STRUCT_001) ──────────────────────────────────────────────────

STRUCT_THRESHOLD: float = 10_000.0
# US Bank Secrecy Act CTR reporting threshold.

STRUCT_BAND_LOW: float = 0.80
# Qualifying lower bound (inclusive): amount_paid >= STRUCT_THRESHOLD * STRUCT_BAND_LOW

STRUCT_BAND_HIGH: float = 0.99
# Qualifying upper bound (exclusive): amount_paid < STRUCT_THRESHOLD * STRUCT_BAND_HIGH

STRUCT_MIN_COUNT: int = 3
# Minimum qualifying outbound transactions in the rolling window to fire.

STRUCT_WINDOW_HOURS: int = 24
# Rolling window in hours (boundary inclusive: span <= STRUCT_WINDOW_HOURS fires).

# ── Pass-through / Rapid In-Out (PASSTHROUGH_001) ────────────────────────────

PASSTHROUGH_MIN_RATIO: float = 0.80
# Fraction of inbound amount that must be paid out within the window.

PASSTHROUGH_WINDOW_HOURS: int = 24
# Outbound must arrive strictly before PASSTHROUGH_WINDOW_HOURS after inbound.
# At exactly PASSTHROUGH_WINDOW_HOURS the rule does NOT fire.

# ── Fan-In / Fan-Out (FAN_IN_001, FAN_OUT_001) ───────────────────────────────

FAN_N: int = 5
# Minimum unique counterparties in the rolling window to fire.

FAN_WINDOW_HOURS: int = 24
# Rolling window in hours (boundary inclusive).

# ── High-Risk Corridor (CORRIDOR_001) ────────────────────────────────────────

HIGH_RISK_COUNTRIES: frozenset[str] = frozenset({
    # FATF Blacklist (June 2023)
    "MM", "IR",
    # FATF Grey List (June 2023)
    "AL", "BB", "BF", "CM", "CD", "GI", "HT", "JM", "JO", "KE",
    "ML", "MZ", "NA", "NG", "PA", "PH", "SN", "SS", "SY", "TZ",
    "TT", "UG", "VU", "VN", "YE", "ZA",
    # FinCEN / OFAC priority jurisdictions
    "PK", "KP", "AF", "IQ", "LB", "LY", "SO", "SD",
    "CU", "VE",
})
# frozenset: immutable at runtime; prevents accidental in-process mutation.

# ── Performance guard ─────────────────────────────────────────────────────────

MAX_WINDOW_ROWS: int = 10_000
# Accounts with more transactions are trimmed to the most-recent MAX_WINDOW_ROWS
# rows before rule evaluation to cap latency on hub accounts.

# ── Step 5 anomaly model (frozen here as single source of truth) ─────────────

ANOMALY_CONTAMINATION: float = 0.005
ANOMALY_FLAGGING_PERCENTILE: float = 0.995

# ── Future rules — not implemented in Phase 1–3 ──────────────────────────────

CYCLE_MAX_LEN: int = 6
BIPARTITE_WINDOW_HOURS: int = 48
VELOCITY_N: int = 20
VELOCITY_WINDOW_HOURS: int = 24
