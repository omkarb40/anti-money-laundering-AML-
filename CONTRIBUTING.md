# Contributing Guide

## Environment Setup

```bash
git clone <repo-url>
cd aml-copilot

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Base install (Steps 0–8, Phase 2, test runner)
pip install -e ".[dev]"

# Phase 3 framework comparison (adds crewai, openai-agents)
pip install -e ".[dev,compare]"
```

Python 3.11+ required. 3.12 is also tested.

---

## Dependency Groups

| Extra | Packages added | When needed |
|---|---|---|
| (none) | polars, rapidfuzz, faker, pydantic, numpy, lxml, langgraph | Steps 0–8, Phase 2 |
| `[dev]` | pytest, pytest-cov | Running tests |
| `[compare]` | crewai, openai-agents | Phase 3 framework comparison |

The base install intentionally excludes `crewai` and `openai-agents`. CI verifies
that `crewai` is absent from the base environment.

---

## Test Categories and Markers

| Marker | Meaning | Skip condition |
|---|---|---|
| *(none)* | Offline unit tests; synthetic fixtures only | Never skipped |
| `integration` | Requires generated artifacts (`eval.jsonl`, `results.jsonl`) | Use `-k "not integration"` |
| `live` | Requires real API keys or network access | Use `-k "not live"` |
| `slow` | Long wall-clock time (full 90-case framework run) | Use `-k "not slow"` |
| `compare` | Phase 3 comparison-specific tests | Use `-k "not compare"` for base suite |

**Standard CI command (no raw data needed):**

```bash
pytest -k "not integration and not live" -q
```

**Phase 3 offline command with coverage gate:**

```bash
pytest tests/test_phase3_compare.py \
    -k "not integration and not live" \
    --cov=aml_copilot.phase3_compare \
    --cov-fail-under=85 -q
```

---

## Formatting and Type Expectations

- No type annotations are required for test helper functions, but all public
  module APIs in `src/` should carry type hints.
- Keep line length ≤ 100 characters where practical.
- Comments should explain *why*, not *what*. Avoid multi-line comment blocks.
- No external formatter is enforced; match the surrounding style.

---

## Frozen Artifact Rules

These files must not be modified after their freeze step:

| File | Frozen at | Consequence of modification |
|---|---|---|
| `data/fixtures/ground_truth_matches.csv` | Step 1 | Sanctions matcher precision becomes unmeasurable |
| `src/aml_copilot/step4_rules/thresholds.py` | Step 4 | Threshold changes after eval = rigging |
| `data/fixtures/eval.jsonl` | Step 6 | Eval changes after baseline = leakage |
| `artifacts/metrics_baseline.json` | Step 8 | Phase 4 control row; must be immutable |

`artifacts/checksums.sha256` records SHA-256 digests of all four. Any modification
causes `run_baseline.py` to abort. Use `git checkout --` to restore a frozen file.

**No decision-threshold tuning after eval construction.** If a threshold change is
needed, the entire pipeline must be rebuilt from Step 4 onward, with a fresh eval
set and a new baseline freeze.

---

## Adding a Fourth Framework

1. Create `src/aml_copilot/phase3_compare/my_runner.py`.
2. Implement the `AMLAgentRunner` protocol:

```python
class MyRunner:
    framework_name: str = "my_framework"

    def run(
        self,
        eval_path: Path,
        baseline_path: Path,
    ) -> list[Phase3CaseResult]:
        ...
```

3. Inside `run()`, call `load_eval_cases(eval_path)`,
   `load_baseline_results(baseline_path)`, and `build_evidence(baseline)` from
   `_shared.py`. Pass the evidence dict to `mock_llm_call`.

4. Register in `RUNNER_REGISTRY` in `run_comparison.py`:

```python
RUNNER_REGISTRY = [
    (LangGraphRunner,    _PKG / "langgraph_runner.py"),
    (CrewAIRunner,       _PKG / "crewai_runner.py"),
    (OpenAIAgentsRunner, _PKG / "openai_agents_runner.py"),
    (MyRunner,           _PKG / "my_runner.py"),  # add here
]
```

5. Add required parity tests:

- Framework produces the same dispositions as the existing three under `mock_llm_call`.
- All 90 (or 5-case mini) eval cases have results.
- `tokens_used == 0` and `cost_usd == 0.0` for every result.
- `comparison_passed` is True when the new framework is included.

6. Add the new dependency to `[compare]` in `pyproject.toml`.

---

## No-Network Offline Requirement

All Phase 3 tests must work without network access, API keys, or environment
variables. `mock_llm_call` provides the decision policy deterministically.

Never add a `requests`, `httpx`, or SDK client call inside a Phase 3 adapter
without a corresponding offline injection mechanism.

---

## Commit Hygiene

- Commit only source files, tests, documentation, and small committed fixtures.
- Do not commit: `data/raw/`, `data/processed/`, `artifacts/results.jsonl`,
  `artifacts/phase2_langgraph_results.jsonl`, `artifacts/phase3_*_results.jsonl`,
  `.env`, `.coverage`, `coverage.xml`.
- Do not commit frozen artifacts with modifications.
- Do not commit CI output JSON files to `artifacts/` unless they represent an
  official 90-case run.
- Keep commits focused. One logical change per commit.
