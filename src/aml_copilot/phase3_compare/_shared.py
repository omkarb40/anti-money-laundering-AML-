"""
Shared Phase 3 runner infrastructure — single source of truth.

Exports
-------
EXPECTED_EVAL_SIZE          : int
load_eval_cases             : (Path) -> list[EvalCase]
load_baseline_results       : (Path) -> dict[str, CaseResult]
build_evidence              : (CaseResult) -> dict[str, Any]
validate_phase3_results     : (list[Phase3CaseResult], list[EvalCase], str) -> None

All three Phase 3 adapters (LangGraph, CrewAI, OpenAI Agents SDK) import
from this module.  Nothing in this module contains framework logic.

──────────────────────────────────────────────────────────────────────────────
LATENCY METHODOLOGY  (enforced by every Phase 3 adapter)
──────────────────────────────────────────────────────────────────────────────
Per-case latency measures framework execution only.

The timer starts immediately before the framework invocation call and stops
immediately after:

  LangGraph   : graph.invoke(initial_state)
  CrewAI      : crew.kickoff()            (via _execute_crew)
  OpenAI Agents: Runner.run_sync(agent, input, …)

Included in the timer
  · framework invocation / kickoff / run_sync
  · framework orchestration overhead
  · tool / node execution
  · output parsing and conversion to the framework decision type

Excluded from the timer
  · loading eval or baseline files
  · constructing reusable runner-level objects (graph, model, agent)
  · one-time graph compilation
  · static prompt / schema / tool construction
  · evidence preparation (build_evidence)
  · result assembly and Pydantic validation
  · metrics computation and file writes

For CrewAI, DeterministicCrewAILLM, Agent, Task, and Crew are constructed
before the timer starts.  crew.kickoff() is the only timed operation.
For OpenAI Agents SDK, DeterministicAMLModel and Agent are constructed once
in OpenAIAgentsRunner.__init__; Runner.run_sync is the only timed operation.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from aml_copilot.schemas import CaseResult, EvalCase, Phase3CaseResult

# ── Constants ─────────────────────────────────────────────────────────────────

EXPECTED_EVAL_SIZE: int = 90


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_eval_cases(eval_path: Path) -> list[EvalCase]:
    """Load and parse all EvalCase rows from eval.jsonl."""
    cases: list[EvalCase] = []
    with open(eval_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(EvalCase.model_validate_json(line))
    return cases


def load_baseline_results(baseline_path: Path) -> dict[str, CaseResult]:
    """Load Phase 1 baseline CaseResult rows, keyed by case_id."""
    results: dict[str, CaseResult] = {}
    with open(baseline_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = CaseResult.model_validate_json(line)
                results[r.case_id] = r
    return results


# ── Evidence construction ─────────────────────────────────────────────────────

def build_evidence(baseline: CaseResult) -> dict[str, Any]:
    """
    Serialise a baseline CaseResult into the shared evidence dict consumed by mock_llm_call.

    Returns exactly five keys:
        sanctions_hits       list[dict]   serialised SanctionsHit objects
        rule_firings         list[dict]   serialised RuleFiring objects
        anomaly_score        dict | None  serialised AnomalyScore or None
        baseline_disposition str          CaseResult.disposition
        baseline_reason      str          CaseResult.decision_reason

    All three Phase 3 adapters call this function, guaranteeing that every
    framework passes an identical evidence dict to mock_llm_call.
    """
    baseline_dict = json.loads(baseline.model_dump_json())
    return {
        "sanctions_hits":       baseline_dict.get("sanctions_hits", []),
        "rule_firings":         baseline_dict.get("rule_firings", []),
        "anomaly_score":        baseline_dict.get("anomaly_score"),
        "baseline_disposition": baseline_dict.get("disposition"),
        "baseline_reason":      baseline_dict.get("decision_reason"),
    }


# ── Protocol invariant validator ──────────────────────────────────────────────

def validate_phase3_results(
    results: list[Phase3CaseResult],
    eval_cases: list[EvalCase],
    framework_name: str,
) -> None:
    """
    Assert all AMLAgentRunner protocol invariants and cost/token constraints.

    Checks (in order)
    -----------------
    1. len(results) == len(eval_cases)
    2. No duplicate case_ids in results
    3. All r.framework == framework_name
    4. All r.case_id and r.account_id are non-empty
    5. r.tokens_used == 0 for every result
    6. r.cost_usd == 0.0 for every result
    7. Every eval case_id has exactly one result (no missing cases)
    8. No result case_id absent from the eval set (no extra cases)

    Raises
    ------
    RuntimeError
        Naming the violated invariant and the offending case IDs.
    """
    expected_count = len(eval_cases)
    if len(results) != expected_count:
        raise RuntimeError(
            f"{framework_name}: produced {len(results)} results; "
            f"expected {expected_count}"
        )

    case_ids = [r.case_id for r in results]
    dup_counts = Counter(case_ids)
    dups = [cid for cid, n in dup_counts.items() if n > 1]
    if dups:
        raise RuntimeError(f"{framework_name}: duplicate case_ids: {dups}")

    wrong = [r.case_id for r in results if r.framework != framework_name]
    if wrong:
        raise RuntimeError(
            f"{framework_name}: {len(wrong)} results carry wrong framework tag "
            f"(expected {framework_name!r}): {wrong[:5]}"
        )

    empty_fields = [r.case_id for r in results if not r.case_id or not r.account_id]
    if empty_fields:
        raise RuntimeError(
            f"{framework_name}: missing required fields in: {empty_fields}"
        )

    bad_tokens = [r.case_id for r in results if r.tokens_used != 0]
    if bad_tokens:
        raise RuntimeError(
            f"{framework_name}: non-zero tokens_used in: {bad_tokens}"
        )

    bad_cost = [r.case_id for r in results if r.cost_usd != 0.0]
    if bad_cost:
        raise RuntimeError(
            f"{framework_name}: non-zero cost_usd in: {bad_cost}"
        )

    eval_ids = {c.case_id for c in eval_cases}
    result_ids = set(case_ids)

    missing = eval_ids - result_ids
    if missing:
        raise RuntimeError(
            f"{framework_name}: {len(missing)} eval cases have no result: "
            f"{sorted(missing)[:5]}"
        )

    extra = result_ids - eval_ids
    if extra:
        raise RuntimeError(
            f"{framework_name}: {len(extra)} result case_ids not in eval set: "
            f"{sorted(extra)[:5]}"
        )
