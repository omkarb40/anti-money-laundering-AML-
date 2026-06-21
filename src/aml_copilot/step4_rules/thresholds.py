# Domain constants — frozen before eval set construction.
# SHA-256 of this file is recorded in artifacts/checksums.sha256.
# Do not modify after freeze.

STRUCT_THRESHOLD: float = 10_000.0   # reporting threshold in dataset currency units
STRUCT_MIN_COUNT: int = 3             # minimum transactions to trigger structuring rule
STRUCT_WINDOW_HOURS: int = 24
STRUCT_BAND_LOW: float = 0.80         # lower bound: txn must be >= 80% of STRUCT_THRESHOLD
STRUCT_BAND_HIGH: float = 0.99        # upper bound: txn must be < 99% of STRUCT_THRESHOLD

PASSTHROUGH_WINDOW_HOURS: int = 24
PASSTHROUGH_MIN_RATIO: float = 0.80   # fraction of inbound that must be forwarded out

FAN_N: int = 5                        # unique counterparties to trigger fan rule
FAN_WINDOW_HOURS: int = 72

CYCLE_MAX_LEN: int = 6                # max cycle length to detect

BIPARTITE_WINDOW_HOURS: int = 48

HIGH_RISK_COUNTRIES: frozenset[str] = frozenset({
    "IR",  # Iran
    "KP",  # North Korea
    "SY",  # Syria
    "CU",  # Cuba
    "VE",  # Venezuela
})

VELOCITY_N: int = 20
VELOCITY_WINDOW_HOURS: int = 24

ANOMALY_CONTAMINATION: float = 0.005  # IsolationForest contamination parameter
ANOMALY_FLAGGING_PERCENTILE: float = 0.995
