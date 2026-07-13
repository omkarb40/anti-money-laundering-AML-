"""
Phase 3 M3 — CrewAI adapter satisfying AMLAgentRunner.

Executes a genuine 3-agent sequential CrewAI Crew for each EvalCase.
The offline DeterministicCrewAILLM delegates all decision logic to
mock_llm_call (the single source of truth in phase3_compare.mock_llm).
Zero network calls; no external API keys required.

Latency methodology
-------------------
Per-case timer wraps crew.kickoff() only (via _execute_crew).
DeterministicCrewAILLM, Agent, Task, and Crew objects are constructed
before the timer starts (via _build_crew).  See phase3_compare._shared
for the full latency contract shared across all Phase 3 adapters.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM

from aml_copilot.phase3_compare._shared import (
    EXPECTED_EVAL_SIZE,
    build_evidence,
    load_baseline_results,
    load_eval_cases,
    validate_phase3_results,
)
from aml_copilot.phase3_compare.mock_llm import MockLLMOutput, mock_llm_call
from aml_copilot.schemas import CaseResult, EvalCase, Phase3CaseResult

_FRAMEWORK: str = "crewai"


class DeterministicCrewAILLM(BaseLLM):
    """Offline custom LLM — zero network calls, delegates to mock_llm_call.

    Evidence is passed as a JSON string in the ``evidence_json`` field so that
    each per-case LLM instance carries its own context without mutable shared
    state.  CrewAI calls ``call()`` directly (bypassing LiteLLM) because we
    subclass ``BaseLLM``.
    """

    model: str = "aml-offline"
    evidence_json: str = ""

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ) -> str:
        evidence: dict[str, Any] = (
            json.loads(self.evidence_json) if self.evidence_json else {}
        )
        result: MockLLMOutput = mock_llm_call(evidence)
        # Return ReAct-style response so CrewAI's agent executor accepts it.
        return f"Thought: AML analysis complete.\nFinal Answer: {json.dumps(result)}"


class CrewAIRunner:
    """AMLAgentRunner adapter for CrewAI (Phase 3 M3).

    Runs a 3-agent sequential Crew per EvalCase.  Per-case objects
    (DeterministicCrewAILLM, Agent, Task, Crew) are constructed before
    the latency timer starts; only crew.kickoff() is timed.

    Memory, planning, delegation, and reasoning are disabled to keep
    execution deterministic and offline.
    """

    framework_name: str = _FRAMEWORK

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    def run(
        self,
        eval_path: Path,
        baseline_path: Path,
    ) -> list[Phase3CaseResult]:
        """Execute the CrewAI AML agent on all eval cases.

        Parameters
        ----------
        eval_path : Path
            data/fixtures/eval.jsonl — 90 frozen EvalCase rows.
        baseline_path : Path
            artifacts/results.jsonl — 90 Phase 1 baseline CaseResult rows.

        Returns
        -------
        list[Phase3CaseResult]
            90 results in eval-file order, all tagged framework="crewai".

        Raises
        ------
        RuntimeError
            If result count, uniqueness, or framework-tag invariants fail.
        """
        if not eval_path.exists():
            raise FileNotFoundError(f"Eval set not found: {eval_path}")
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Baseline results not found: {baseline_path}\n"
                "Run python -m aml_copilot.step7_runner.run_baseline first."
            )

        cases: list[EvalCase] = load_eval_cases(eval_path)
        baseline_map: dict[str, CaseResult] = load_baseline_results(baseline_path)

        results: list[Phase3CaseResult] = []

        for case in cases:
            baseline = baseline_map.get(case.case_id)
            if baseline is None:
                raise RuntimeError(
                    f"Case {case.case_id!r} absent from baseline results"
                )

            evidence = build_evidence(baseline)

            # PREPARE: construct LLM, agents, tasks, and crew before the timer.
            llm = DeterministicCrewAILLM(
                model="aml-offline",
                evidence_json=json.dumps(evidence),
            )
            crew = _build_crew(llm, verbose=self._verbose)

            # EXECUTE: timer wraps only crew.kickoff() and output parsing.
            t0 = time.perf_counter()
            llm_output = _execute_crew(crew, evidence)
            latency_ms = (time.perf_counter() - t0) * 1000

            agent_disposition: Literal["ESCALATE", "CLEAR"] = llm_output["disposition"]
            agent_override = agent_disposition != baseline.disposition

            results.append(
                Phase3CaseResult(
                    framework=_FRAMEWORK,
                    case_id=case.case_id,
                    account_id=case.account_id,
                    disposition=agent_disposition,
                    decision_reason=llm_output["decision_reason"],
                    sanctions_hits=baseline.sanctions_hits,
                    rule_firings=baseline.rule_firings,
                    anomaly_score=baseline.anomaly_score,
                    latency_ms=latency_ms,
                    agent_reasoning=llm_output["reasoning"],
                    agent_override=agent_override,
                    baseline_disposition=baseline.disposition,
                    human_review_flagged=llm_output["human_review"],
                    tokens_used=0,
                    cost_usd=0.0,
                )
            )

        validate_phase3_results(results, cases, _FRAMEWORK)
        return results


# ── Internal helpers ──────────────────────────────────────────────────────────


def _build_crew(llm: DeterministicCrewAILLM, verbose: bool = False) -> Crew:
    """
    Build a 3-agent sequential Crew with the given LLM.

    Called before the latency timer.  Returns a ready-to-kickoff Crew;
    does not invoke any network or LLM calls.
    """
    analyst = Agent(
        role="Evidence Analyst",
        goal="Parse and structure AML case evidence for risk assessment",
        backstory=(
            "Specializes in parsing financial transaction evidence, identifying "
            "sanctions hits, rule firings, and anomaly scores."
        ),
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        tools=[],
        memory=None,
        planning=False,
        reasoning=False,
    )
    specialist = Agent(
        role="Risk Decision Specialist",
        goal="Apply AML risk policy to determine case disposition",
        backstory=(
            "Expert in anti-money laundering typologies and FATF standards. "
            "Applies a structured decision policy to sanctions, rule, and anomaly evidence."
        ),
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        tools=[],
        memory=None,
        planning=False,
        reasoning=False,
    )
    finalizer = Agent(
        role="Case Finalizer",
        goal="Produce the final structured AML case result",
        backstory=(
            "Ensures AML case results are properly formatted with all required "
            "fields for downstream compliance reporting."
        ),
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        tools=[],
        memory=None,
        planning=False,
        reasoning=False,
    )

    t1 = Task(
        description="Analyze the AML case evidence and prepare a structured evidence summary.",
        expected_output="A structured evidence summary for the decision specialist.",
        agent=analyst,
    )
    t2 = Task(
        description="Apply AML risk policy to the evidence and produce a disposition decision.",
        expected_output=(
            "A JSON object with keys: disposition, decision_reason, "
            "reasoning, confidence, human_review."
        ),
        agent=specialist,
        context=[t1],
    )
    t3 = Task(
        description="Finalize and validate the AML case disposition decision.",
        expected_output="Final validated AML case disposition as a JSON object.",
        agent=finalizer,
        context=[t2],
    )

    return Crew(
        agents=[analyst, specialist, finalizer],
        tasks=[t1, t2, t3],
        process=Process.sequential,
        verbose=verbose,
        memory=False,
        planning=False,
    )


def _execute_crew(crew: Crew, evidence: dict[str, Any]) -> MockLLMOutput:
    """
    Invoke crew.kickoff() and parse the output.

    Called inside the latency timer.  Falls back to a direct mock_llm_call
    if CrewAI's agent executor appends unexpected text that breaks JSON parsing
    (DeterministicCrewAILLM always produces valid JSON under normal operation).
    """
    crew_output = crew.kickoff()
    raw = crew_output.raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = mock_llm_call(evidence)

    return parsed  # type: ignore[return-value]


def _build_evidence(baseline: CaseResult) -> dict[str, Any]:
    """Thin wrapper for backward compatibility; delegates to build_evidence."""
    return build_evidence(baseline)


def _run_crew(evidence: dict[str, Any], verbose: bool = False) -> MockLLMOutput:
    """
    Convenience wrapper used by unit tests.

    Builds a fresh Crew with the given evidence and executes it in one call.
    Not used by CrewAIRunner.run() (which separates prepare and execute for
    correct latency measurement).
    """
    llm = DeterministicCrewAILLM(
        model="aml-offline",
        evidence_json=json.dumps(evidence),
    )
    crew = _build_crew(llm, verbose=verbose)
    return _execute_crew(crew, evidence)
