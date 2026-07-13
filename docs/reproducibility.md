# Reproducibility Guide

This document describes how to reproduce the full baseline — from a fresh clone
with raw data to a verified `metrics_baseline.json` — in eight steps.

Expected wall-clock time on a modern laptop: **10–20 minutes** (dominated by
OFAC fuzzy screening over 515 K accounts in Step 7; Step 5 robust-z scoring
completes in ~2 seconds).

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | 3.12 also tested |
| pip | latest | `pip install --upgrade pip` |
| ~2 GB free RAM | — | Feature matrix (float32, 515 K × 15 features) + transaction graph |
| ~2 GB disk (raw data) | — | HI-Small CSV + OFAC XML |

---

## 1 — Clone and install

```bash
git clone <repo-url>
cd aml-copilot

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

---

## 2 — Acquire raw data

### IBM AMLSim HI-Small

Download from one of:
- [IBM AMLSim GitHub](https://github.com/IBM/AMLSim) (see `README` → datasets section)
- [IEEE DataPort](https://ieee-dataport.org/open-access/amlsim) (requires free account)

Extract and place:
```
data/raw/HI-Small_Trans.csv        # 5,078,345 rows
data/raw/HI-Small_Patterns.txt
```

Verify the download with:
```bash
wc -l data/raw/HI-Small_Trans.csv
# Expected: 5078346 (including header)
```

### OFAC SDN + Consolidated Lists

Download **Advanced XML format** from:
- https://ofac.treasury.gov/specially-designated-nationals-and-blocked-persons-list-sdn-list/sdn-advanced-data-formats

Place:
```
data/raw/ofac/sdn_advanced.xml
data/raw/ofac/cons_advanced.xml
```

> The lists are updated regularly.  For exact reproducibility, use the snapshot
> from the date the ground-truth fixture was built.  The fixture was created with
> the SDN list published in mid-2025.  Using a much newer list may add entries
> that affect hard-negative score bounds.

---

## 3 — Configure environment

```bash
cp .env.example .env
# .env uses the default paths; edit only if your layout differs
```

The defaults assume data is under `data/raw/` relative to the repo root, which
matches the layout above.

---

## 4 — Run Steps 0–6 (data preparation)

Steps 0–6 must be run once to build the processed data and frozen fixtures.  If
`data/fixtures/eval.jsonl` and `data/fixtures/ground_truth_matches.csv` are
already present (they are committed to the repo), Steps 1 and 6 do not need to
be re-run.

```bash
# Step 0: validate the dataset (assertion gate)
python -m aml_copilot.step0_scaffold.data_loader

# Step 1: assign synthetic names + build ground-truth fixture
# (skip if data/fixtures/ground_truth_matches.csv already exists)
python -m aml_copilot.step1_identity.build

# Step 2: no standalone runner — OFAC index is built at startup of Step 7

# Step 3: no standalone runner — graph is built at startup of Step 7

# Step 4: no standalone runner — rules fire per-case inside Step 7

# Step 5: no standalone runner — anomaly model fits at startup of Step 7

# Step 6: build the 90-case eval set
# (skip if data/fixtures/eval.jsonl already exists and checksums pass)
python -m aml_copilot.step6_eval.builder
```

After Step 6, verify the frozen fixtures:
```bash
python -m aml_copilot.utils.checksum --verify artifacts/checksums.sha256
# Expected: [OK] All checksums verified
```

---

## 5 — Run the baseline (Step 7)

This is the main pipeline run.  All tools are initialized once at startup; then
each of the 90 eval cases is processed end-to-end.

```bash
python -m aml_copilot.step7_runner.run_baseline \
    --eval data/fixtures/eval.jsonl \
    --out  artifacts/results.jsonl
```

Expected output:
```
================================================================
results.jsonl: artifacts/results.jsonl  (90 cases)
  Dispositions:   {'CLEAR': 43, 'ESCALATE': 47}
  Reasons:        {'clear': 43, 'sanctions_or_critical_rule': 47}
  Latency p50:    ~51 ms
  Latency p95:    ~57 ms
================================================================
```

> Disposition counts are **fully deterministic** — identical across machines and
> runs (robust-z anomaly scoring has no random state). Latency values are
> hardware-dependent.

---

## 6 — Compute and freeze metrics (Step 8)

```bash
python -m aml_copilot.step8_metrics.metrics
```

On first run this writes `artifacts/metrics_baseline.json` and appends its
SHA-256 to `artifacts/checksums.sha256`.

On subsequent runs (when the file already exists) it prints the frozen report
and exits without modifying anything:
```
[SKIP] artifacts/metrics_baseline.json already frozen. Use --force to rebuild.
```

Expected metrics:
```
Disposition accuracy:       0.755556
False-clear rate (wtd):     0.225166   ← PRIMARY
Sanctions precision:        1.000000
Sanctions recall:           1.000000
Latency p50:                ~51 ms     (hardware-dependent)
Latency p95:                ~57 ms     (hardware-dependent)
Total cost:                 $0.00
```

---

## 7 — Final checksum verification

```bash
python -m aml_copilot.utils.checksum --verify artifacts/checksums.sha256
```

All four entries should pass:
```
[OK] All checksums verified: artifacts/checksums.sha256
```

---

## 7b — Phase 3 Framework Comparison (optional; requires Step 7 output)

After completing Step 7 (which produces `artifacts/results.jsonl`), install the
compare extras and run the three-framework comparison:

```bash
pip install -e ".[dev,compare]"

python -m aml_copilot.phase3_compare.run_comparison \
    --eval      data/fixtures/eval.jsonl \
    --baseline  artifacts/results.jsonl \
    --out       artifacts/phase3_comparison_metrics.json
```

Expected output:

```
VERDICT: PASS
PASS — All frameworks produce identical results.
```

**Offline five-case demo (no raw data required):**

```bash
python -m aml_copilot.phase3_compare.run_comparison \
    --eval      tests/fixtures/phase3_mini_eval.jsonl \
    --baseline  tests/fixtures/phase3_mini_baseline.jsonl \
    --out       /tmp/phase3_mini_comparison.json
```

**Validate the comparison artifact:**

```python
from aml_copilot.schemas import Phase3ComparisonMetrics
import json

data = json.loads(open("artifacts/phase3_comparison_metrics.json").read())
cm = Phase3ComparisonMetrics.model_validate(data)
assert cm.comparison_passed
assert cm.eval_size == 90
assert cm.all_dispositions_agree
print("Validation OK:", cm.frameworks[0].disposition_accuracy)
```

---

## 8 — Run the test suite

```bash
# Fast unit + Phase 3 offline tests (no raw data, no API keys required)
pytest -k "not integration and not live" -q

# Phase 3 comparison tests with coverage gate
pytest tests/test_phase3_compare.py \
    -k "not integration and not live" \
    --cov=aml_copilot.phase3_compare \
    --cov-fail-under=85 -q

# Full suite including integration tests (requires raw data and generated artifacts)
pytest -q
```

Most integration tests are automatically skipped when raw data or built artifacts
are absent (they use `pytest.mark.integration`). Use `-k "not integration and not live"`
to run all offline tests cleanly on a fresh clone.

---

## Expected File State After Full Reproduction

```
artifacts/
  checksums.sha256                     # 4 entries, all verified
  metrics_baseline.json                # frozen Phase 1 control row (FROZEN)
  phase2_langgraph_metrics.json        # Phase 2/3 summary (committed)
  phase3_comparison_metrics.json       # Phase 3 comparison summary (committed)
  results.jsonl                        # 90 CaseResult rows (not committed)
data/
  fixtures/
    eval.jsonl                         # 90 EvalCase rows (committed, FROZEN)
    ground_truth_matches.csv           # 50-row sanctions fixture (committed, FROZEN)
  processed/
    accounts.parquet                   # 515,080 accounts (not committed)
    identity_overlay.parquet           # 515,080 accounts with names (not committed)
  raw/                                 # not committed — provide your own
    HI-Small_Trans.csv
    HI-Small_Patterns.txt
    ofac/
      sdn_advanced.xml
      cons_advanced.xml
```

---

---

## Determinism and Reproducibility Limits

**Fully deterministic (bitwise-identical across runs):**
- All dispositions (ESCALATE / CLEAR)
- Disposition accuracy and false-clear rate
- Sanctions precision and recall
- Override rate and human-review rate
- Agreement flags (all dispositions agree, etc.)

**Not bitwise-identical (expected to vary):**
- `generated_at` in JSON artifacts — this is a UTC timestamp set at write time
- Per-case `latency_ms` — wall-clock timing is machine-specific
- `latency_p50_ms`, `latency_p95_ms`, `average_latency_ms` — derived from latency

When validating reproducibility semantically, compare all fields except
`generated_at` and latency-derived fields. The `model_dump()` comparison in tests
uses `exclude={"generated_at", "latency_p50_ms", "latency_p95_ms"}` where needed.

---

## Frozen Artifact Policy

| Artifact | Status | Why |
|---|---|---|
| `data/fixtures/ground_truth_matches.csv` | Committed + checksummed | Must not change after sanctions matcher tuning |
| `src/aml_copilot/step4_rules/thresholds.py` | Committed + checksummed | Must not change after eval construction |
| `data/fixtures/eval.jsonl` | Committed + checksummed | Modifying after seeing baseline = eval leakage |
| `artifacts/metrics_baseline.json` | Committed + checksummed | Phase 4 control row; immutable reference |
| `artifacts/phase2_langgraph_metrics.json` | Committed (not checksummed) | Compact summary; exploratory Phase 3 artifact |
| `artifacts/phase3_comparison_metrics.json` | Committed (not checksummed) | Official Phase 3 comparison summary |
| `artifacts/results.jsonl` | Not committed | Machine-specific latency; fully reproducible |
| `artifacts/phase2_langgraph_results.jsonl` | Not committed | 107 KB per-case results; not needed for reproducibility |

Phase 3 artifacts (`phase3_comparison_metrics.json`) are **not** added to
`artifacts/checksums.sha256` because they are exploratory records (generated at
research time), not integrity gates for the production pipeline.

---

## CI Equivalence

| Local command | CI workflow | Step |
|---|---|---|
| `pytest -k "not integration and not live and not compare" -q` | `test.yml` | "Run base tests" |
| `pytest tests/test_phase3_compare.py -k "not integration and not live" --cov-fail-under=85` | `phase3-compare.yml` | "Run offline Phase 3 comparison tests" |
| `python -m aml_copilot.phase3_compare.run_comparison --eval tests/fixtures/phase3_mini_eval.jsonl ...` | `phase3-compare.yml` | "Run mini comparison CLI" |
| `python -m aml_copilot.utils.checksum --verify artifacts/checksums.sha256` | `test.yml` | "Verify frozen artifact checksums" |

---

## Checksum Portability

`artifacts/checksums.sha256` stores **repo-relative paths** (e.g.
`data/fixtures/eval.jsonl`) so the manifest is portable across clone locations
and operating systems.  `verify_checksums()` resolves paths relative to the
package root at runtime.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Frozen file missing: data/fixtures/eval.jsonl` | Fixture not present | Run Step 6 or check out the committed file |
| `Checksum mismatch for …/thresholds.py` | Thresholds were edited | Restore from git: `git checkout -- src/aml_copilot/step4_rules/thresholds.py` |
| `Expected 5078345 rows, got N` | Wrong dataset (HI-Medium) | Re-download HI-Small specifically |
| `OFAC index has 0 entries` | XML schema version mismatch | Verify the XML file is `sdn_advanced.xml` (not `sdn.csv`) |
| OOM during Step 7 | <2 GB free RAM | Close other applications; Step 5 uses float32 feature matrix (~350 MB); transaction graph is the main consumer |
