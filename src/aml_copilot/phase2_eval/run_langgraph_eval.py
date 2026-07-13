"""
Phase 2.5 — LangGraph agent evaluation, mocked-LLM mode.

Architecture
------------
A three-node LangGraph state machine processes each EvalCase using
pre-computed tool outputs from the deterministic baseline (results.jsonl).
No raw transaction data is required.

Nodes
~~~~~
  prepare_evidence  →  llm_decide  →  finalize

1. prepare_evidence  — parses the baseline CaseResult JSON into a structured
   evidence block (sanctions_hits, rule_firings, anomaly_score).

2. llm_decide  — the mock LLM.  Policy (extends the baseline decision table):
     Branch 1 (unchanged):
       - any SanctionsHit.match_score ≥ 0.90 → ESCALATE
       - any RuleFiring.severity == 3         → ESCALATE
     Branch 2 (agent extension):
       - anomaly_percentile ≥ 0.90 AND max_rule_severity ≥ 2 → ESCALATE
         Baseline requires is_flagged (99.5th pct); agent lowers the effective
         threshold to 90th pct when paired with elevated rule evidence.
     Branch 3: → CLEAR  (human review flagged if pct > 0.85)

   The deterministic mock policy lives in phase3_compare.mock_llm.  Real-provider
   integration should replace or inject the decision provider at the framework
   runner layer rather than editing the backward-compatible alias in this module.

3. finalize  — no-op; extensible for post-processing or tool-call injection.

Inputs (committed to repo — no raw data required)
-------------------------------------------------
  data/fixtures/eval.jsonl       90 EvalCase rows
  artifacts/results.jsonl        90 CaseResult rows (baseline tool outputs)

Outputs
-------
  artifacts/phase2_langgraph_results.jsonl   (not frozen — Phase 2.5 artifact)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from aml_copilot.schemas import (
    AnomalyScore,
    CaseResult,
    EvalCase,
    RuleFiring,
    SanctionsHit,
)
from aml_copilot.phase3_compare.mock_llm import (
    mock_llm_call,
    AGENT_ANOMALY_PCT_THRESHOLD,        # re-exported: test_phase2_eval imports this
    AGENT_MIN_RULE_SEV_FOR_OVERRIDE,    # re-exported: test_phase2_eval imports this
    HUMAN_REVIEW_ANOMALY_THRESHOLD,     # re-exported: test_phase2_eval imports this
    HUMAN_REVIEW_ANOMALY_MIN,           # re-exported: new M1 alias
    HUMAN_REVIEW_ANOMALY_MAX,           # re-exported: new M1 alias
)
from aml_copilot.phase3_compare._shared import (
    EXPECTED_EVAL_SIZE,
    build_evidence as _build_evidence,
    load_eval_cases as _load_eval_cases,          # re-exported: test_phase2_eval imports this
    load_baseline_results as _load_baseline_results,  # re-exported: test_phase2_eval imports this
)
_mock_llm_call = mock_llm_call          # alias: test_phase2_eval imports _mock_llm_call

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parents[3]

_DEFAULTS: dict[str, Path] = {
    "eval":     _ROOT / "data/fixtures/eval.jsonl",
    "baseline": _ROOT / "artifacts/results.jsonl",
    "out":      _ROOT / "artifacts/phase2_langgraph_results.jsonl",
}

# EXPECTED_EVAL_SIZE imported from phase3_compare._shared (single source of truth)


# ── Phase2 result schema ──────────────────────────────────────────────────────

class Phase2CaseResult(BaseModel):
    """CaseResult extended with Phase 2.5 agent fields."""
    case_id: str
    account_id: str
    disposition: Literal["ESCALATE", "CLEAR"]
    decision_reason: str
    sanctions_hits: list[SanctionsHit]
    rule_firings: list[RuleFiring]
    anomaly_score: Optional[AnomalyScore] = None
    latency_ms: float
    # Phase 2.5 additions
    agent_reasoning: str
    agent_override: bool
    baseline_disposition: Literal["ESCALATE", "CLEAR"]
    human_review_flagged: bool
    tokens_used: int = 0
    cost_usd: float = 0.0


# ── LangGraph state ───────────────────────────────────────────────────────────

class AMLAgentState(TypedDict):
    case_id: str
    account_id: str
    case_type: str
    notes: str
    baseline_result_json: str        # serialised CaseResult
    evidence: dict[str, Any]         # parsed by prepare_evidence
    agent_disposition: str           # "ESCALATE" | "CLEAR" | ""
    agent_decision_reason: str
    agent_reasoning: str
    agent_confidence: float
    human_review_flagged: bool
    tokens_used: int
    cost_usd: float


# ── LangGraph nodes ───────────────────────────────────────────────────────────

def prepare_evidence_node(state: AMLAgentState) -> dict[str, Any]:
    """Parse baseline result JSON into a structured evidence block."""
    baseline = CaseResult.model_validate_json(state["baseline_result_json"])
    return {"evidence": _build_evidence(baseline)}


def llm_decide_node(state: AMLAgentState) -> dict[str, Any]:
    """Invoke the (mock) LLM; populate disposition and reasoning fields."""
    result = mock_llm_call(state["evidence"])
    return {
        "agent_disposition": result["disposition"],
        "agent_decision_reason": result["decision_reason"],
        "agent_reasoning": result["reasoning"],
        "agent_confidence": result["confidence"],
        "human_review_flagged": result["human_review"],
        "tokens_used": 0,
        "cost_usd": 0.0,
    }


def finalize_node(state: AMLAgentState) -> dict[str, Any]:
    """No-op finalizer. Extensible for post-processing or tool calls."""
    return {}


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph():
    """Build and compile the AML LangGraph agent."""
    g: StateGraph = StateGraph(AMLAgentState)
    g.add_node("prepare_evidence", prepare_evidence_node)
    g.add_node("llm_decide", llm_decide_node)
    g.add_node("finalize", finalize_node)
    g.set_entry_point("prepare_evidence")
    g.add_edge("prepare_evidence", "llm_decide")
    g.add_edge("llm_decide", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


# ── Public runner API ─────────────────────────────────────────────────────────

def run(
    eval_path: Path,
    baseline_path: Path,
    out_path: Path,
) -> list[Phase2CaseResult]:
    """
    Run the LangGraph agent on all eval cases.

    Parameters
    ----------
    eval_path : Path
        data/fixtures/eval.jsonl — 90 EvalCase rows.
    baseline_path : Path
        artifacts/results.jsonl — 90 deterministic baseline CaseResult rows.
    out_path : Path
        Destination for phase2_langgraph_results.jsonl.

    Returns
    -------
    list[Phase2CaseResult]
        One result per eval case; 90 rows total.
    """
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval set not found: {eval_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline results not found: {baseline_path}\n"
            "Run python -m aml_copilot.step7_runner.run_baseline first."
        )

    cases = _load_eval_cases(eval_path)
    baseline_map = _load_baseline_results(baseline_path)

    graph = build_graph()
    results: list[Phase2CaseResult] = []

    for i, case in enumerate(cases, start=1):
        if i % 10 == 0 or i == 1:
            logger.info("  Case %d / %d  [%s]", i, len(cases), case.case_id)

        baseline_result = baseline_map.get(case.case_id)
        if baseline_result is None:
            raise RuntimeError(
                f"Case {case.case_id!r} absent from baseline results"
            )

        initial_state: AMLAgentState = {
            "case_id": case.case_id,
            "account_id": case.account_id,
            "case_type": case.case_type,
            "notes": case.notes,
            "baseline_result_json": baseline_result.model_dump_json(),
            "evidence": {},
            "agent_disposition": "",
            "agent_decision_reason": "",
            "agent_reasoning": "",
            "agent_confidence": 0.0,
            "human_review_flagged": False,
            "tokens_used": 0,
            "cost_usd": 0.0,
        }

        t0 = time.perf_counter()
        final_state = graph.invoke(initial_state)
        latency_ms = (time.perf_counter() - t0) * 1000

        agent_disposition: Literal["ESCALATE", "CLEAR"] = final_state["agent_disposition"]
        agent_override = agent_disposition != baseline_result.disposition

        results.append(
            Phase2CaseResult(
                case_id=case.case_id,
                account_id=case.account_id,
                disposition=agent_disposition,
                decision_reason=final_state["agent_decision_reason"],
                sanctions_hits=baseline_result.sanctions_hits,
                rule_firings=baseline_result.rule_firings,
                anomaly_score=baseline_result.anomaly_score,
                latency_ms=latency_ms,
                agent_reasoning=final_state["agent_reasoning"],
                agent_override=agent_override,
                baseline_disposition=baseline_result.disposition,
                human_review_flagged=final_state["human_review_flagged"],
                tokens_used=final_state["tokens_used"],
                cost_usd=final_state["cost_usd"],
            )
        )

    if len(results) != len(cases):
        raise RuntimeError(
            f"Result count mismatch: produced {len(results)}, expected {len(cases)}"
        )

    # Atomic write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(
        "\n".join(r.model_dump_json() for r in results) + "\n",
        encoding="utf-8",
    )
    tmp.replace(out_path)
    logger.info("[Phase2] phase2_langgraph_results.jsonl written: %s", out_path)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2.5 LangGraph agent evaluation (mocked-LLM mode)"
    )
    p.add_argument("--eval",     default=str(_DEFAULTS["eval"]))
    p.add_argument("--baseline", default=str(_DEFAULTS["baseline"]))
    p.add_argument("--out",      default=str(_DEFAULTS["out"]))
    return p.parse_args()


def _print_summary(results: list[Phase2CaseResult]) -> None:
    from collections import Counter

    import numpy as np

    dispositions = Counter(r.disposition for r in results)
    overrides = sum(1 for r in results if r.agent_override)
    reviews = sum(1 for r in results if r.human_review_flagged)
    latencies = np.array([r.latency_ms for r in results])

    print(f"\n{'=' * 64}")
    print(f"Phase 2.5 LangGraph agent — {len(results)} cases")
    print(f"  Dispositions:      {dict(sorted(dispositions.items()))}")
    print(f"  Baseline overrides:{overrides:3d} / {len(results)} ({overrides/len(results):.1%})")
    print(f"  Human review flags:{reviews:3d} / {len(results)} ({reviews/len(results):.1%})")
    print(f"  Latency p50:       {np.percentile(latencies, 50):.2f} ms")
    print(f"  Latency p95:       {np.percentile(latencies, 95):.2f} ms")
    print(f"  Total tokens:      0  (mock mode)")
    print(f"  Total cost:        $0.00  (mock mode)")
    print(f"{'=' * 64}\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    try:
        results = run(
            eval_path=Path(args.eval),
            baseline_path=Path(args.baseline),
            out_path=Path(args.out),
        )
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)

    _print_summary(results)
    print("[Phase2] run_langgraph_eval complete.")


if __name__ == "__main__":
    main()
