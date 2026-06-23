# AML Investigation Copilot — IBM AMLSim HI-Small

A deterministic, zero-cost anti-money laundering triage pipeline built over the
IBM AMLSim HI-Small synthetic dataset. The system screens accounts against OFAC
sanctions, resolves transaction-graph entity chains, fires typology rules, and
produces anomaly scores. A fixed-precedence decision table routes each case to
**ESCALATE** or **CLEAR**.

This is the **Phase 1–3 control baseline** (no LLM). Phase 4 will add an LLM
layer evaluated against this frozen baseline.

---

## Baseline Metrics (90-case eval set)

| Metric | Value |
|---|---|
| Disposition accuracy | 75.56 % |
| Weighted false-clear rate — primary | 22.52 % |
| Sanctions precision | 100.0 % |
| Sanctions recall | 100.0 % |
| Latency p50 | ~51 ms |
| Latency p95 | ~57 ms |
| Total LLM cost | $0.00 |

> Latency measured on a 2024 MacBook Pro M3. Accuracy and precision/recall are
> hardware-independent. All disposition counts and metrics are **fully
> deterministic** — identical results on every run.

**Sanctions screening is saturated** at 100 % precision and recall across the
15 true-positive and 15 hard-negative eval cases. Future accuracy gains are
expected to come from conflict-resolution and typology cases (the remaining 60
eval cases), not from further tuning of the sanctions matcher.

---

## Architecture — Eight Steps

| Step | Module | Role |
|---|---|---|
| 0 | `step0_scaffold` | Load and validate HI-Small CSV; assert row/account/ratio counts |
| 1 | `step1_identity` | Assign synthetic Faker names; build 50-row ground-truth fixture |
| 2 | `step2_sanctions` | OFAC SDN/Consolidated parse + fuzzy-name screening (Jaro-Winkler + token sort) |
| 3 | `step3_entity` | Transaction-graph traversal → hop-1 / hop-2 counterparty chain |
| 4 | `step4_rules` | Eight typology rules (structuring, passthrough, fan-out, cycle, …) |
| 5 | `step5_anomaly` | Deterministic robust-z anomaly scoring; explicit feature-leakage exclusion |
| 6 | `step6_eval` | Assemble frozen 90-case eval set across five case types (write-once) |
| 7 | `step7_runner` | Fixed-precedence decision table + baseline CLI runner |
| 8 | `step8_metrics` | Read-only metric computation; freeze `metrics_baseline.json` |

See [`docs/architecture.md`](docs/architecture.md) for detailed module descriptions,
data-flow diagram, and Pydantic schema overview.

### Fully Deterministic Baseline

Every component of the Phase 1 pipeline is deterministic:

- **Step 1** — Faker names assigned from a committed seed (`FAKER_SEED=42`).
- **Steps 2–4** — Rule evaluation is purely threshold-based; OFAC fuzzy matching
  uses a fixed algorithm (Jaro-Winkler + token sort ratio).
- **Step 5** — Robust-z anomaly scoring uses median and MAD statistics computed
  from the population. No random state, no bootstrap, no model fitting.
  `score_accounts()` produces bitwise-identical output on every call.
- **Steps 6–8** — Eval set is frozen; decision table is immutable; metrics are
  a pure function of results and gold labels.

The only hardware-dependent value is wall-clock latency.  Disposition counts,
accuracy, and precision/recall are identical across machines and runs.

### Phase Roadmap

| Phase | Description |
|---|---|
| **1 (current)** | Deterministic AML baseline: sanctions screening, entity resolution, rule engine, robust-z anomaly scoring, fixed-precedence decision table |
| **2** | LangGraph agent: conflict resolution, typology interpretation, human-in-the-loop review, SAR narrative generation |
| **3** | Framework comparison: LangGraph vs CrewAI vs OpenAI Agents SDK |
| **4** | LLM evaluation against the frozen Phase 1 baseline |
| **5** | Deployment and production architecture |

---

## Dataset Requirements

**Raw data is not included in this repository.** Download it separately before
running the pipeline.

### IBM AMLSim HI-Small

Available from the [IBM AMLSim repository](https://github.com/IBM/AMLSim)
or the [IEEE DataPort release](https://ieee-dataport.org/open-access/amlsim).

Required files — place in `data/raw/`:

| File | Rows | Description |
|---|---|---|
| `HI-Small_Trans.csv` | 5,078,345 | Transaction records |
| `HI-Small_Patterns.txt` | — | Typology pattern labels |

> **Assertion gate:** the pipeline asserts exact row count (5,078,345 ± 0)
> and account count (~515,080 ± 100). Using HI-Medium or a truncated download
> fails immediately.

### OFAC SDN + Consolidated Lists (Advanced XML format)

Available from [U.S. Treasury OFAC](https://ofac.treasury.gov/specially-designated-nationals-and-blocked-persons-list-sdn-list/sdn-advanced-data-formats).

Required files — place in `data/raw/ofac/`:

- `sdn_advanced.xml`
- `cons_advanced.xml`

OFAC data is a U.S. government publication and is in the public domain.
Always use the current list from the [official OFAC site](https://ofac.treasury.gov)
for any compliance application.

---

## Setup

```bash
git clone <repo-url>
cd aml-copilot

# Create a virtual environment (Python 3.11+ required)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install the package and dev dependencies
pip install -e ".[dev]"

# Copy and fill in the environment template
cp .env.example .env
# Edit .env to set DATA_DIR and OFAC paths if non-default
```

---

## Running Tests

Most unit tests use small synthetic fixtures and **do not require the raw dataset**:

```bash
# Fast unit tests — no raw data required
pytest tests/ -k "not integration" -q

# Full suite — requires HI-Small and OFAC data in data/raw/
pytest -q
```

---

## Running the Baseline Pipeline

Steps must be executed in order. See [`docs/reproducibility.md`](docs/reproducibility.md)
for the complete step-by-step sequence with expected outputs.

```bash
# Step 7 — run the full baseline decision table
python -m aml_copilot.step7_runner.run_baseline \
    --eval data/fixtures/eval.jsonl \
    --out  artifacts/results.jsonl

# Step 8 — compute and freeze baseline metrics
python -m aml_copilot.step8_metrics.metrics

# Verify all frozen artifact checksums
python -m aml_copilot.utils.checksum --verify artifacts/checksums.sha256
```

---

## Committed Frozen Artifacts

These files are committed to the repository and must not be modified after creation.
Their SHA-256 digests are recorded in `artifacts/checksums.sha256` and verified
at the start of every pipeline run.

| File | Frozen at | Contents |
|---|---|---|
| `data/fixtures/ground_truth_matches.csv` | Step 1 | 50-row sanctions ground truth (20 TP + 30 hard negatives) |
| `data/fixtures/eval.jsonl` | Step 6 | 90-case evaluation set across five case types |
| `src/aml_copilot/step4_rules/thresholds.py` | Step 4 | Rule thresholds (must not change after eval construction) |
| `artifacts/checksums.sha256` | Incrementally | SHA-256 manifest for all frozen files |
| `artifacts/metrics_baseline.json` | Step 8 | Phase 4 control row |

**`artifacts/results.jsonl` is not committed** — it is a generated output that
changes with every run and carries machine-specific latency timings.

### Fixture tracking decision

`data/fixtures/eval.jsonl` (26 KB) and `ground_truth_matches.csv` (4.6 KB) are
tracked because:
- They are synthetic (Faker names + AMLSim account IDs, no real personal data).
- The OFAC canonical names in `ground_truth_matches.csv` are from the public-domain
  U.S. Treasury SDN list.
- Tracking them allows tests and checksum verification to work on a fresh clone
  without rebuilding Steps 1 and 6 from raw data.

---

## Responsible Use

This project uses **synthetic transaction data** (IBM AMLSim) and publicly
available **government sanctions lists** (OFAC SDN/Consolidated). It is a
research baseline and has not been validated for production AML compliance.

- Do not use this system as the sole or primary basis for any real financial
  compliance decision.
- The OFAC sanctions lists used here are maintained by the U.S. Department of
  the Treasury. Always retrieve current data from the
  [official OFAC site](https://ofac.treasury.gov) for compliance applications.
- This codebase is released under the MIT License. See [`LICENSE`](LICENSE).
