# AML Investigation Copilot

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-788%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-~89%25-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Overview

This repository is a reproducible AI engineering research project built over the IBM AMLSim HI-Small synthetic dataset (~5 million transactions, ~515,000 accounts) and the U.S. Treasury OFAC sanctions lists. It implements a complete anti-money laundering triage pipeline — sanctions screening, entity graph resolution, typology rule evaluation, and anomaly scoring — wrapped in a fixed-precedence decision table that routes each case to ESCALATE or CLEAR without any LLM calls.

The project's central question is whether agent orchestration frameworks can improve AML case triage while preserving auditability and measurable cost/latency. Answering that requires a control baseline held constant while the orchestration layer varies. Phases 1–3 build that baseline by implementing the same deterministic decision policy in three frameworks (LangGraph, CrewAI, OpenAI Agents SDK) and verifying they produce identical results on a frozen 90-case evaluation set.

Reproducibility is a first-class constraint here, not an afterthought. Four artifacts are SHA-256 checksummed at the moment they are created — ground-truth fixtures, rule thresholds, the eval set, and the baseline metrics — and the pipeline refuses to run if any of them have changed. This prevents the most common evaluation errors: threshold rigging, eval leakage, and retroactive fixture adjustment.

---

## Features

- Sanctions screening via OFAC SDN + Consolidated lists (fuzzy Jaro-Winkler + token-sort-ratio, AKA expansion)
- Transaction-graph entity resolution (2-hop traversal, 515K-node graph, hop-2 capped at 50)
- Eight AML typology rules: structuring, layering, fan-out/in, cycle, bipartite, corridor, velocity
- Robust-z anomaly scoring — fully deterministic, no model object, no random state, no label leakage
- Fixed-precedence decision table with explainable typed evidence on every disposition
- Framework comparison harness: LangGraph, CrewAI, and OpenAI Agents SDK under a shared policy
- Frozen 90-case evaluation set with reproducible SHA-256 checksums
- Offline mini fixtures (5 cases covering all policy branches) for zero-dependency testing
- GitHub Actions CI: base test suite + Phase 3 comparison workflow
- 788 tests, ~89% coverage, offline by default

---

## System Architecture

```
IBM AMLSim HI-Small                  OFAC SDN + Consolidated
(~5M transactions, ~515K accounts)   (public XML)
              │                                │
              ▼                                ▼
       Step 0: Scaffold               Step 2: OFAC Index
       Step 1: Identity Overlay       (AKA expansion, normalization)
              │                                │
              ▼                                │
       Sanctions Screening ◄───────────────────┘
              │
              ▼
       Entity Resolution
       (2-hop graph traversal, pattern labels)
              │
              ▼
       AML Rules + Risk Signals
       (8 typology rules · robust-z anomaly score)
              │
              ▼
       Evidence Aggregation
       (SanctionsHit + RuleFiring + AnomalyScore → typed Pydantic bundle)
              │
              ▼
       Agent Orchestration
       ├── LangGraph Runner (70 LOC)
       ├── CrewAI Runner (247 LOC)
       └── OpenAI Agents Runner (376 LOC)
              │
              ▼
       Explainable Investigation Decision
       (ESCALATE / CLEAR + decision_reason + human_review_flagged)
```

Full architecture diagram (Mermaid): [`docs/images/phase3_architecture.mmd`](docs/images/phase3_architecture.mmd)

Detailed module descriptions: [`docs/architecture.md`](docs/architecture.md)

---

## Framework Comparison

All three framework adapters run against the same 90 frozen eval cases using the same shared decision policy (`mock_llm_call`). The policy is a deterministic 4-branch function; no LLM API is called.

| Framework | LOC | Approach | p50 latency |
|---|---|---|---|
| **LangGraph** | 70 | TypedDict state · conditional edge routing | 0.78 ms |
| **CrewAI** | 247 | Agent/Task/Crew · sequential execution | 43 ms |
| **OpenAI Agents SDK** | 376 | `Runner.run_sync` · function tools | 3.8 ms |

All adapters satisfy the `AMLAgentRunner` structural protocol: `framework_name: str` + `run(eval_path, baseline_path) -> list[Phase3CaseResult]`.

**What this comparison measures and what it does not:**

- Every framework receives identical pre-computed evidence bundles
- Every framework executes the same deterministic decision policy
- This is **not** a comparison of LLM reasoning quality — no model is called
- Equal accuracy across frameworks is expected and confirms that the adapters correctly wrap the shared policy
- The comparison isolates orchestration overhead, implementation complexity, and latency methodology

See [`docs/phase3_framework_comparison.md`](docs/phase3_framework_comparison.md) for the experimental design, decision matrix, and portfolio claims.

---

## Evaluation Results

All values are read directly from committed artifacts.

| Metric | Value |
|---|---|
| Tests | 788 passing |
| Coverage | ~89% |
| Phase 1 accuracy (decision table only) | 75.56% |
| Phase 1 weighted false-clear rate | 22.52% |
| Shared-policy accuracy (all frameworks) | 78.89% |
| Shared-policy weighted false-clear rate | 17.22% |
| Sanctions precision / recall | 100% / 100% |
| Disposition agreement across frameworks | 90/90 |
| Reasoning agreement across frameworks | 90/90 |
| Frameworks compared | 3 (LangGraph, CrewAI, OpenAI Agents SDK) |
| API calls | 0 |
| Token cost | $0.00 |
| Offline reproducibility | Full (mini fixtures committed) |

The accuracy improvement from 75.56% to 78.89% comes from the shared policy's additional conflict-resolution branch, not from any framework choice. All three frameworks produce identical dispositions by design.

Source files: `artifacts/metrics_baseline.json` (Phase 1) and `artifacts/phase3_comparison_metrics.json` (frameworks).

---

## Repository Structure

```
aml-copilot/
├── pyproject.toml
├── CONTRIBUTING.md
├── .env.example
├── data/
│   ├── raw/                          # gitignored — download separately
│   ├── processed/                    # gitignored — regenerated by pipeline
│   └── fixtures/                     # committed and FROZEN
│       ├── eval.jsonl                # 90 EvalCase rows
│       └── ground_truth_matches.csv  # 50-row sanctions ground truth
├── src/aml_copilot/
│   ├── schemas.py                    # all Pydantic types (single source of truth)
│   ├── utils/
│   │   ├── checksum.py               # SHA-256 freeze verification
│   │   └── normalize.py              # Unicode NFKD normalization
│   ├── step0_scaffold/ … step8_metrics/   # Phase 1 pipeline (8 steps)
│   ├── phase2_eval/                  # LangGraph Phase 2 adapter + evaluator
│   └── phase3_compare/               # three framework adapters + comparison runner
│       ├── _shared.py                # shared I/O, evidence builder, validator
│       ├── mock_llm.py               # shared deterministic decision policy
│       ├── protocol.py               # AMLAgentRunner structural protocol
│       ├── langgraph_runner.py
│       ├── crewai_runner.py
│       ├── openai_agents_runner.py
│       ├── metrics.py                # per-framework and comparison metric computation
│       ├── run_comparison.py         # CLI entrypoint
│       └── testing.py                # offline mini-fixture helpers (test use only)
├── tests/
│   ├── fixtures/                     # committed offline test fixtures
│   │   ├── phase3_mini_eval.jsonl    # 5-case mini eval (all policy branches)
│   │   ├── phase3_mini_baseline.jsonl
│   │   └── phase3_expected_comparison.json
│   └── test_*.py
├── artifacts/
│   ├── checksums.sha256              # SHA-256 of frozen artifacts
│   ├── metrics_baseline.json         # Phase 1 baseline (FROZEN)
│   ├── phase2_langgraph_metrics.json # Phase 2 summary
│   └── phase3_comparison_metrics.json
├── docs/
│   ├── architecture.md
│   ├── reproducibility.md
│   ├── phase3_framework_comparison.md
│   └── images/phase3_architecture.mmd
└── .github/workflows/
    ├── test.yml                      # base suite CI
    └── phase3-compare.yml            # Phase 3 comparison CI
```

---

## Quick Start

### Install

```bash
git clone <repo-url>
cd aml-copilot

python -m venv .venv && source .venv/bin/activate

# Base install (Steps 0–8, LangGraph)
pip install -e ".[dev]"

# Add CrewAI and OpenAI Agents SDK for Phase 3 comparison
pip install -e ".[dev,compare]"
```

### Run the test suite

```bash
# Offline tests — no raw data, no API keys required
pytest -k "not integration and not live" -q

# With coverage
pytest -k "not integration and not live" --cov=aml_copilot --cov-report=term-missing -q

# Full suite (requires raw data and generated artifacts)
pytest -q
```

### Run the mini comparison (no raw data required)

Works immediately after cloning. Covers all four decision-policy branches.

```bash
python -m aml_copilot.phase3_compare.run_comparison \
  --eval      tests/fixtures/phase3_mini_eval.jsonl \
  --baseline  tests/fixtures/phase3_mini_baseline.jsonl \
  --out       /tmp/phase3_mini_comparison.json
```

Expected output:
```
VERDICT: PASS
PASS — All frameworks produce identical results.
```

### Run the full 90-case comparison (requires raw data)

Prerequisites: `data/raw/HI-Small_Trans.csv` from IBM AMLSim and `data/raw/ofac/sdn_advanced.xml` + `cons_advanced.xml` from U.S. Treasury OFAC.

```bash
# Phase 1: run the deterministic baseline
python -m aml_copilot.step7_runner.run_baseline \
    --eval data/fixtures/eval.jsonl \
    --out  artifacts/results.jsonl

# Phase 3: compare all three frameworks
python -m aml_copilot.phase3_compare.run_comparison \
    --eval      data/fixtures/eval.jsonl \
    --baseline  artifacts/results.jsonl \
    --out       artifacts/phase3_comparison_metrics.json

# Verify frozen artifact checksums
python -m aml_copilot.utils.checksum --verify artifacts/checksums.sha256
```

---

## Documentation

| Document | Description |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Module boundaries, data flow, step-by-step design decisions |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Step-by-step sequence from fresh clone; checksum policy; known non-determinism |
| [`docs/phase3_framework_comparison.md`](docs/phase3_framework_comparison.md) | Experimental design, decision matrix, adapter implementation notes, portfolio claims |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Development workflow, test conventions, frozen-artifact policy |

---

## Current Capabilities

- Sanctions screening against OFAC SDN and Consolidated lists with AKA expansion and Unicode NFKD normalization
- Fuzzy name matching (Jaro-Winkler + token-sort-ratio) with a calibrated 0.85/0.90 threshold ladder
- Entity graph construction and 2-hop counterparty traversal across 515K accounts
- Eight AML typology rules: structuring, rapid in/out passthrough, fan-out, fan-in, cycle, bipartite scatter-gather, high-risk corridor, and general velocity
- Deterministic robust-z anomaly scoring with an explicit feature-exclusion list and no label leakage
- Fixed-precedence decision table producing `ESCALATE` / `CLEAR` with typed evidence on every case
- Three equivalent framework adapters (LangGraph, CrewAI, OpenAI Agents SDK) verified to produce identical output
- 90-case frozen evaluation set with five case-type slices (IBM-labeled, sanctions hits, near-misses, rule/anomaly conflicts, typology)
- SHA-256 checksum enforcement on four frozen artifacts; pipeline aborts on any mismatch
- Offline mini fixtures for CI and local development with no raw-data dependency
- GitHub Actions workflows for base tests and Phase 3 comparison

---

## Limitations

- **Synthetic data.** IBM AMLSim generates transactions algorithmically. Real financial crime patterns differ in distribution, noise, and typology complexity.
- **No live LLM reasoning.** The current decision policy is a deterministic 4-branch function. Replacing it with a real model (future work) changes the experiment fundamentally.
- **Framework comparison is deterministic.** All three adapters call the same function. Equal accuracy is expected and does not reflect LLM reasoning quality or framework capability in general.
- **Static OFAC snapshot.** Sanctions lists change. The ground-truth fixture was built against a mid-2025 SDN snapshot; hard-negative score bounds may shift on list update.
- **Machine-specific latency.** p50/p95 latency values in committed artifacts reflect a MacBook Pro M3. Your hardware will differ.
- **Not for production compliance.** AML typology rules here do not constitute a compliance program. The system has not been tested against real customer records or validated for regulatory requirements.
- **Human review remains required.** The `human_review_flagged` field is a routing signal, not a final decision. All escalated cases require investigator review.

---

## Future Work

- Replace `mock_llm_call` with real LLM providers and measure accuracy, false-clear rate, cost, and latency against the frozen Phase 1 baseline
- Evaluate multiple models within each framework to separate model quality from orchestration overhead
- Expand explainability: natural-language rationale generation anchored to the typed evidence bundle
- Collect human-investigator feedback to calibrate the decision table and measure inter-rater agreement
- Investigate production deployment considerations: latency SLAs, audit logging, model versioning, and compliance documentation

---

## Design Principles

**Reproducibility.** Four artifacts are checksummed at creation and enforced on every pipeline run. Seeded Faker output, committed thresholds, and frozen eval sets make every result independently verifiable.

**Deterministic evaluation.** No randomness enters the scoring, policy, or framework execution paths. Two independent runs on the same inputs must produce byte-identical outputs.

**Explainability.** Every disposition carries a typed evidence bundle — `SanctionsHit`, `RuleFiring`, `AnomalyScore` — and a `decision_reason` string naming the exact branch of the decision table that fired. Nothing is black-box.

**Framework neutrality.** The `AMLAgentRunner` structural protocol ensures all three adapters are interchangeable. The comparison harness validates orchestration differences in isolation; it does not favor any framework.

**Testability.** 788 tests, offline mini fixtures, and four pytest markers (`integration`, `live`, `slow`, `compare`) let contributors run the full suite without raw data or API keys. Coverage is enforced in CI.

**Modular architecture.** Steps 2–5 have no imports from each other. All inter-step data passes through `schemas.py` Pydantic types. Module boundaries are documented in `docs/architecture.md` and enforced by code review.

---

## Data Attribution

**IBM AMLSim HI-Small**

> Altman, Erik and Blanuša, Jovan and von Niederhäusern, Luc and Bhatt, Bharat and
> Sperl, Erik and Stockinger, Kurt. "Realistic Synthetic Financial Transactions for
> Anti-Money Laundering Models." *Advances in Neural Information Processing Systems*
> (NeurIPS), 2023.

Available from the [IBM AMLSim repository](https://github.com/IBM/AMLSim) and [IEEE DataPort](https://ieee-dataport.org/open-access/amlsim).

**OFAC SDN + Consolidated Lists**

Published by the U.S. Department of the Treasury, Office of Foreign Assets Control. Public domain. Always retrieve the current list from the [official OFAC site](https://ofac.treasury.gov) for any compliance application.

---

## Responsible Use

This project uses synthetic transaction data and public government sanctions lists. It is a research baseline and has not been validated for production AML compliance.

- Do not use this system as the sole or primary basis for any real financial compliance decision.
- Released under the MIT License. See [`LICENSE`](LICENSE) for the full text.
