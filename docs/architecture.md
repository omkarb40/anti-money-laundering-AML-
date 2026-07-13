# Architecture — AML Investigation Copilot

## Overview

The pipeline is a sequence of eight independent steps connected by typed
[Pydantic v2](https://docs.pydantic.dev/) data structures defined in
`src/aml_copilot/schemas.py`.  Each step owns a narrow slice of logic and never
imports from adjacent steps — all inter-step data flows through schema types.

```
HI-Small_Trans.csv          OFAC SDN/Consolidated XML
        │                           │
        ▼                           ▼
   [Step 0: loader]          [Step 2: index]
   accounts.parquet          ofac_index (in-memory)
        │                           │
        ▼                           │
   [Step 1: overlay]               │
   identity_overlay.parquet        │
   ground_truth_matches.csv ◄──────┘   (FROZEN)
        │
        ├──────────────────────────────────────┐
        │                                      │
        ▼                                      ▼
   [Step 2: screen]            [Step 3: resolve]
   List[SanctionsHit]          EntityChain
        │                                      │
        │            [Step 4: engine]          │
        │            List[RuleFiring] ◄────────┤
        │                    │                 │
        │            [Step 5: scorer]          │
        │            AnomalyScore ◄────────────┘
        │                    │
        └────────┬───────────┘
                 ▼
        [Step 7: decision table]
        CaseResult {disposition, evidence}
                 │
                 ▼
        [Step 8: metrics]
        MetricsReport → metrics_baseline.json (FROZEN)
```

---

## Module Boundary Rules

| Module | Owns | Never imports from |
|---|---|---|
| `step0_scaffold` | Load, validate, cache raw CSV | Any downstream step |
| `step1_identity` | Name assignment + fixture construction | Screening logic |
| `step2_sanctions` | OFAC index + fuzzy matching | Transaction data |
| `step3_entity` | Graph traversal, counterparty chain | Scoring or disposition |
| `step4_rules` | Deterministic threshold evaluation | Anomaly model, OFAC |
| `step5_anomaly` | Deterministic feature + robust-z score | Rule thresholds, labels |
| `step6_eval` | Eval set assembly (write-once) | Pipeline execution |
| `step7_runner` | Orchestration + decision table | Metric computation |
| `step8_metrics` | Read-only metric computation | Any data mutation |

Steps 2–5 have **no imports from each other**.

---

## Step-by-Step Description

### Step 0 — Scaffold

Loads `HI-Small_Trans.csv` with Polars (lazy evaluation for memory efficiency)
and derives the account universe as the union of `From Account ID` and
`To Account ID`.

**Assertion gate** (fails loudly on wrong dataset):
- Row count: 5,078,345 ± 0
- Account count: ~515,080 ± 100
- `Is_Laundering` ratio: 0.0008–0.0012

These assertions distinguish HI-Small from HI-Medium (32 M rows / 2.08 M
accounts) and catch truncated downloads.

**Key output:** `accounts.parquet`, in-memory Polars DataFrame.

---

### Step 1 — Synthetic Identity Overlay

Assigns a synthetic Faker name to every account using a committed seed (default
`FAKER_SEED=42`) so output is deterministic. Then constructs a 50-row
ground-truth fixture for sanctions evaluation.

**Ground-truth composition:**

| Flavor | Count | Score expectation |
|---|---|---|
| Exact match | 5 | == 1.00 |
| Transliteration (NFKD) | 5 | ≥ 0.90 |
| Typo / OCR error | 5 | ≥ 0.85 |
| Partial reorder | 5 | token_sort_ratio ≥ 0.90 |
| Hard negative | 30 | JW 0.80–0.88, never ≥ 0.90 |

Hard negatives (HNs) are built in the **same pass** as positives using the real
OFAC index; they share a surname with a real SDN entry but differ in given name.
Building HNs in the same pass prevents tuning the matcher against them.

**Safety constraint:** real OFAC canonical names appear only in
`ground_truth_matches.csv` (the `ofac_canonical_name` column, for reference) and
in the internal screening index.  They do not appear in
`identity_overlay.parquet`'s `name` column or in `results.jsonl`.

**Key outputs:**
- `identity_overlay.parquet` — 515,080 rows, columns: `account_id`, `name`,
  `country`, `kyc_risk`
- `data/fixtures/ground_truth_matches.csv` — **FROZEN** immediately after creation

---

### Step 2 — Sanctions Screen

Matches each account's assigned name against every entry in the OFAC
SDN + Consolidated lists (AKA-expanded, NFKD-normalized).

**Matching pipeline:**
1. Unicode NFKD normalize both strings (é → e, Müller → Muller)
2. Compute Jaro-Winkler similarity
3. Compute `token_sort_ratio / 100` (handles name-component reordering)
4. Final score = `max(jw, token_sort_ratio_normalized)`
5. Emit `SanctionsHit` if score ≥ 0.85

Scores ≥ 0.90 trigger Branch 1 of the decision table (ESCALATE).

**Key output:** `List[SanctionsHit]` per account.

---

### Step 3 — Entity Resolve

Builds an in-memory adjacency dict from all 5 M transaction edges, then does a
bounded breadth-first traversal to collect:
- `hop1_counterparties` — direct transaction neighbours
- `hop2_counterparties` — second-degree neighbours (capped at 50 to prevent
  hub-account explosion)
- `pattern_label` — typology label from `HI-Small_Patterns.txt`, if present

**Key output:** `EntityChain` per account.

---

### Step 4 — Transaction Rules

Eight deterministic rules covering the IBM AMLSim typologies and FATF patterns:

| Rule ID | Typology | Severity |
|---|---|---|
| `STRUCT_001` | Structuring | 3 |
| `PASSTHROUGH_001` | Layering / rapid in-out | 3 |
| `CYCLE_001` | Cycle | 3 |
| `FAN_OUT_001` | Fan-out | 2 |
| `FAN_IN_001` | Fan-in | 2 |
| `BIPARTITE_001` | Scatter-gather | 2 |
| `CORRIDOR_001` | High-risk corridor | 1 |
| `VELOCITY_001` | General velocity | 1 |

All numeric thresholds live in `step4_rules/thresholds.py`, which is **frozen
before the eval set is constructed** so that thresholds cannot be tuned against
eval performance.

**Key output:** `List[RuleFiring]` per account, each with `severity` 1–3.

---

### Step 5 — Anomaly Score

Computes a **deterministic robust-z composite score** for every account using
the full 515 K-account feature matrix.  No model is trained; no random state is
involved.

**Algorithm** (per account, per feature `j`):

```
median_j  = median(X[:, j])
MAD_j     = median(|X[:, j] − median_j|)
scale_j   = 1.4826 × MAD_j   [fallback: std_j + ε when MAD_j = 0]
z_j       = clip(|x_j − median_j| / scale_j, 0, 25)
composite = mean(z_0, …, z_P)          [higher = more anomalous]
```

The `is_flagged` boolean is set when `percentile ≥ ANOMALY_FLAGGING_PERCENTILE`
(top 0.5 % by default).

**Permitted features (non-leaky):**
- Transaction count per 7-day and 30-day window
- Amount statistics (mean, std, max, skewness)
- Counterparty diversity ratio
- Round-number transaction fraction
- Time-of-day entropy
- Weekend / off-hours fraction

**Explicitly excluded** (would leak the `Is_Laundering` label via IBM's data
generation logic): running balance delta, cumulative net flow.  Every run logs
the exclusion list to `AnomalyScore.excluded_features`.

**Key output:** `AnomalyScore` per account — robust-z composite score (higher =
more anomalous), rank-based percentile [0.0, 1.0], and `is_flagged` boolean.
Output is bitwise-identical across runs.

---

### Step 6 — Eval Set (write-once)

Assembles a 90-case evaluation set from real AMLSim accounts across five slices:

| Slice | Count | Source |
|---|---|---|
| IBM-labeled disposition cases | 30 | `Is_Laundering == 1` rows |
| Sanctions hits | 15 | True positives from `ground_truth_matches.csv` |
| Sanctions near-misses | 15 | Hard negatives from `ground_truth_matches.csv` |
| Rules-vs-anomaly conflicts | 10 | Rule fires without anomaly flag (or vice versa) |
| Typology coverage | 20 | 2–3 cases per IBM typology |

Gold labels come from IBM ground truth, never from system output.  The seed is
committed (`EVAL_SEED=42`) so construction is reproducible.

**Key output:** `data/fixtures/eval.jsonl` — **FROZEN immediately after creation.**

---

### Step 7 — Decision Table + Runner

Fixed-precedence decision table (immutable after baseline freeze):

```
IF   any SanctionsHit.score ≥ 0.90
  OR any RuleFiring.severity == 3
THEN ESCALATE  (reason: "sanctions_or_critical_rule")

ELSE IF AnomalyScore.is_flagged == True
     AND any RuleFiring.severity >= 2
THEN ESCALATE  (reason: "anomaly_plus_elevated_rule")

ELSE CLEAR
```

The runner loads all tools once at startup, times each case with
`time.perf_counter`, and writes `results.jsonl` atomically (temp file + rename).
Per-case failures raise loudly with the case ID — no silent CLEAR fallback.

**Key output:** `artifacts/results.jsonl` — 90 `CaseResult` rows (not committed).

---

### Step 8 — Metrics

Reads the frozen eval set and the runner output, computes all metrics via a pure
function (`compute_metrics`), then freezes the result.

**Primary metric: weighted false-clear rate**

A severity-3 false negative (an ESCALATE case cleared) carries 3× the weight of
a severity-1 false negative.  Severity weights: {1: 1×, 2: 2×, 3: 3×}.

Sanctions precision and recall are computed separately using the
`sanctions_hit` / `sanctions_near_miss` eval slices.

**Key output:** `artifacts/metrics_baseline.json` — **FROZEN as the Phase 4 control row.**

---

## Pydantic Schema Summary

All inter-step types are defined in `src/aml_copilot/schemas.py`.

| Schema | Produced by | Consumed by |
|---|---|---|
| `Transaction` | Step 0 | Steps 3, 4 |
| `Account` | Step 0 | Steps 1, 3 |
| `GroundTruthRow` | Step 1 | Step 2 (calibration), Step 6 |
| `SanctionsHit` | Step 2 | Step 7 |
| `EntityChain` | Step 3 | Step 4 |
| `RuleFiring` | Step 4 | Step 7 |
| `AnomalyScore` | Step 5 | Step 7 |
| `EvalCase` | Step 6 | Steps 7, 8 |
| `CaseResult` | Step 7 | Step 8 |
| `MetricsReport` | Step 8 | (frozen artifact) |

---

## Frozen Artifact Integrity

`artifacts/checksums.sha256` records the SHA-256 digest of each frozen file
using **repo-relative paths** so the manifest is portable across clone locations.

`verify_checksums()` is called at the start of every Step 7 and Step 8 run.
Any digest mismatch or missing file aborts with a non-zero exit code.

---

## Technology Stack

| Library | Role |
|---|---|
| [Polars](https://www.pola.rs) | DataFrame operations (lazy eval; required for 5 M-row efficiency) |
| [rapidfuzz](https://github.com/maxbachmann/RapidFuzz) | Jaro-Winkler + `token_sort_ratio` (C extension) |
| [Faker](https://faker.readthedocs.io) | Seeded synthetic name generation |
| [Pydantic v2](https://docs.pydantic.dev) | Schema validation and JSON serialization |
| [NumPy](https://numpy.org) | Robust-z anomaly scoring (median, MAD, ranking) |
| [lxml](https://lxml.de) | OFAC XML parsing |
| [pytest](https://pytest.org) | Unit and integration tests |

---

## Phase 2 — LangGraph Orchestration (Offline Policy)

Phase 2 wraps the Phase 1 evidence pipeline in a LangGraph agent that applies a
shared deterministic policy (`mock_llm_call`). It does not call a real LLM.

The LangGraph adapter is implemented in `src/aml_copilot/phase2_eval/`.
Its metrics are stored in `artifacts/phase2_langgraph_metrics.json`.

**Phase 2 results (90-case eval, offline policy):**

| Metric | Value |
|---|---|
| Disposition accuracy | 78.89% |
| Weighted false-clear rate | 17.22% |
| Override rate (policy vs. baseline) | 5.56% |
| Human-review rate | 16.67% |
| Tokens used | 0 |
| Cost | $0.00 |

---

## Phase 3 — Framework Comparison

Phase 3 implements the same shared policy (`mock_llm_call`) in three agent
frameworks: LangGraph, CrewAI, and OpenAI Agents SDK. The experiment holds policy
and evidence constant and varies only the orchestration framework.

All adapters satisfy the `AMLAgentRunner` protocol (`src/aml_copilot/phase3_compare/protocol.py`).

**Architecture diagram:** see [`docs/images/phase3_architecture.mmd`](images/phase3_architecture.mmd)
for the Mermaid source. Embed it in any Mermaid-capable viewer, or use
[mermaid.live](https://mermaid.live) to render it.

**Detailed comparison:** see [`docs/phase3_framework_comparison.md`](phase3_framework_comparison.md).

**Phase 3 results (90-case eval, all frameworks):**

| Metric | LangGraph | CrewAI | OpenAI Agents SDK |
|---|---|---|---|
| Accuracy | 78.89% | 78.89% | 78.89% |
| LOC | 70 | 247 | 376 |
| Latency p50 (ms) | 0.78 | 42.96 | 3.81 |
| Tokens / Cost | 0 / $0.00 | 0 / $0.00 | 0 / $0.00 |

All three frameworks agree on 90/90 dispositions, decision reasons, and
human-review flags. This is expected: they call the same policy function.

### AMLAgentRunner Protocol

Every Phase 3 adapter implements:

```python
class AMLAgentRunner(Protocol):
    framework_name: str           # class-level string
    def run(
        self,
        eval_path: Path,
        baseline_path: Path,
    ) -> list[Phase3CaseResult]: ...
```

To add a fourth framework, implement this protocol and register the runner in
`RUNNER_REGISTRY` in `src/aml_copilot/phase3_compare/run_comparison.py`.

### Shared Decision Policy (`mock_llm_call`)

All frameworks pass an evidence dict to `mock_llm_call` and receive a deterministic
decision. The four branches (in precedence order):

| Branch | Condition | Disposition | human_review |
|---|---|---|---|
| 1 | Any SanctionsHit score ≥ 0.90 | ESCALATE | False |
| 2 | Any RuleFiring severity == 3 | ESCALATE | False |
| 3 | Anomaly percentile ≥ 0.90 AND any severity ≥ 2 | ESCALATE | True |
| 4 | Fallthrough | CLEAR | True if pct > 0.85, else False |

This policy **intentionally mirrors the Phase 1 decision table** with one addition
(Branch 3's anomaly threshold and human-review flag). It is not a trained model and
does not call any LLM endpoint.

### Canonical Eval Path Safeguard

When `run_comparison.py` is invoked against `data/fixtures/eval.jsonl` (the
canonical frozen eval), it requires exactly 90 cases. When invoked against any other
path (such as the five-case mini fixture in `tests/fixtures/`), it validates only
that result count equals eval count. This prevents accidental production runs against
reduced fixtures while keeping the runners fixture-size-agnostic.

### Mini Fixture

`tests/fixtures/phase3_mini_eval.jsonl` (5 cases) covers all four `mock_llm_call`
policy branches and enables offline testing without raw data, API keys, or the full
90-case run:

```bash
python -m aml_copilot.phase3_compare.run_comparison \
  --eval   tests/fixtures/phase3_mini_eval.jsonl \
  --baseline tests/fixtures/phase3_mini_baseline.jsonl \
  --out    /tmp/phase3_mini_comparison.json
```
