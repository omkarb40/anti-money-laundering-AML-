"""
Phase 3 M4 — OpenAI Agents SDK adapter satisfying AMLAgentRunner.

Architecture
------------
A genuine OpenAI Agents SDK execution path processes each EvalCase:

  1. parse_evidence (function_tool) — receives the pre-built evidence dict from
     build_evidence() (_shared.py) as a JSON string; mirrors the evidence format
     consumed by mock_llm_call so all three frameworks receive identical input.

  2. decide_disposition (function_tool) — calls mock_llm_call(evidence) once;
     the only permitted source of disposition logic.

  3. DeterministicAMLModel.get_response — a custom offline Model implementation
     that issues tool calls in order (step 1, step 2) and converts the result to
     a structured OpenAIAgentDecision on the third call.  Zero HTTP calls; no API key.

Runner.run_sync is the execution entry point.  RunConfig(tracing_disabled=True)
prevents any trace-export network traffic per-run.

Installed SDK
-------------
openai-agents 0.18.2

Model abstract methods implemented
-----------------------------------
  async get_response(system_instructions, input, model_settings, tools,
                     output_schema, handoffs, tracing, *,
                     previous_response_id, conversation_id, prompt)
            -> ModelResponse
  async stream_response(...)   — not used; raises NotImplementedError

Latency methodology
-------------------
DeterministicAMLModel and Agent are constructed once in OpenAIAgentsRunner.__init__
via _create_agent().  Runner.run_sync is the only timed operation per case.
See phase3_compare._shared for the full latency contract shared across all Phase 3 adapters.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel

from agents import (
    Agent,
    Model,
    ModelResponse,
    ModelTracing,
    RunConfig,
    Runner,
    Usage,
    function_tool,
)
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from aml_copilot.phase3_compare._shared import (
    EXPECTED_EVAL_SIZE,
    build_evidence,
    load_baseline_results,
    load_eval_cases,
    validate_phase3_results,
)
from aml_copilot.phase3_compare.mock_llm import MockLLMOutput, mock_llm_call
from aml_copilot.schemas import CaseResult, EvalCase, Phase3CaseResult

_FRAMEWORK: str = "openai_agents"


# ── Internal structured-output model ──────────────────────────────────────────


class OpenAIAgentDecision(BaseModel, extra="forbid"):
    """Structured output produced by the AML Triage Agent per case."""

    disposition: Literal["ESCALATE", "CLEAR"]
    decision_reason: str
    agent_reasoning: str
    human_review_flagged: bool


# ── Function tools ─────────────────────────────────────────────────────────────


@function_tool
def parse_evidence(case_data_json: str) -> str:
    """Parse AML case data and return the shared evidence dict as JSON.

    Receives the output of build_evidence() (_shared.py) as a JSON string
    so that all three Phase 3 frameworks pass identical evidence to mock_llm_call.
    """
    data: dict[str, Any] = json.loads(case_data_json)
    evidence: dict[str, Any] = {
        "sanctions_hits":       data.get("sanctions_hits", []),
        "rule_firings":         data.get("rule_firings", []),
        "anomaly_score":        data.get("anomaly_score"),
        "baseline_disposition": data.get("baseline_disposition"),
        "baseline_reason":      data.get("baseline_reason"),
    }
    return json.dumps(evidence)


@function_tool
def decide_disposition(evidence_json: str) -> str:
    """Apply the shared AML risk policy and return the disposition decision as JSON.

    Delegates exclusively to mock_llm_call — the single permitted source of
    disposition logic in Phase 3.  Contains no independent policy conditions.
    """
    evidence: dict[str, Any] = json.loads(evidence_json)
    result: MockLLMOutput = mock_llm_call(evidence)
    return json.dumps(result)


# ── Offline custom Model ───────────────────────────────────────────────────────


class DeterministicAMLModel(Model):
    """Offline custom Model — zero network calls, zero API key requirement.

    Implements a three-step state machine driven by the tool-call history in the
    ``input`` list that Runner passes on each call:

      Call 1: emit parse_evidence(case_data_json=<user input>)
      Call 2: emit decide_disposition(evidence_json=<parse_evidence output>)
      Call 3: return final structured answer from decide_disposition output

    No state is kept between calls or between cases.
    """

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: Any,
        tools: list[Any],
        output_schema: Any | None,
        handoffs: list[Any],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any | None,
    ) -> ModelResponse:
        usage = Usage(requests=1, input_tokens=0, output_tokens=0, total_tokens=0)
        input_list: list[Any] = input if isinstance(input, list) else []

        # Build named_outputs: tool_name -> tool_output_string
        call_id_to_name: dict[str, str] = {
            item["call_id"]: item["name"]
            for item in input_list
            if isinstance(item, dict) and item.get("type") == "function_call"
        }
        named_outputs: dict[str, str] = {
            call_id_to_name.get(item["call_id"], item["call_id"]): item.get("output", "")
            for item in input_list
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        }

        # Step 1: request parse_evidence
        if "parse_evidence" not in named_outputs:
            user_input: str = next(
                (
                    item["content"]
                    for item in input_list
                    if isinstance(item, dict) and item.get("role") == "user"
                ),
                "",
            )
            return ModelResponse(
                output=[
                    ResponseFunctionToolCall(
                        call_id="call-parse",
                        name="parse_evidence",
                        arguments=json.dumps({"case_data_json": user_input}),
                        type="function_call",
                    )
                ],
                usage=usage,
                response_id=None,
                request_id=None,
            )

        # Step 2: request decide_disposition
        if "decide_disposition" not in named_outputs:
            return ModelResponse(
                output=[
                    ResponseFunctionToolCall(
                        call_id="call-decide",
                        name="decide_disposition",
                        arguments=json.dumps(
                            {"evidence_json": named_outputs["parse_evidence"]}
                        ),
                        type="function_call",
                    )
                ],
                usage=usage,
                response_id=None,
                request_id=None,
            )

        # Step 3: convert decide_disposition output to structured answer
        decision: dict[str, Any] = json.loads(named_outputs["decide_disposition"])
        final = OpenAIAgentDecision(
            disposition=decision["disposition"],
            decision_reason=decision["decision_reason"],
            agent_reasoning=decision["reasoning"],
            human_review_flagged=decision["human_review"],
        )
        text_item = ResponseOutputText(
            text=final.model_dump_json(),
            type="output_text",
            annotations=[],
        )
        msg = ResponseOutputMessage(
            id="msg-final",
            content=[text_item],
            role="assistant",
            status="completed",
            type="message",
        )
        return ModelResponse(
            output=[msg],
            usage=usage,
            response_id=None,
            request_id=None,
        )

    async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        raise NotImplementedError(
            "DeterministicAMLModel does not support streaming"
        )


# ── Runner ─────────────────────────────────────────────────────────────────────


class OpenAIAgentsRunner:
    """AMLAgentRunner adapter for the OpenAI Agents SDK (Phase 3 M4).

    Runs a genuine Agent+Runner execution per EvalCase using:
      - two function_tool callables (parse_evidence, decide_disposition)
      - DeterministicAMLModel — offline custom Model, zero HTTP calls
      - structured output_type=OpenAIAgentDecision per run
      - RunConfig(tracing_disabled=True) — no trace-export network traffic

    DeterministicAMLModel and Agent are constructed once in __init__ via
    _create_agent().  Per-case latency measures only Runner.run_sync().
    """

    framework_name: str = _FRAMEWORK

    def __init__(self, *, tracing_disabled: bool = True) -> None:
        self._tracing_disabled = tracing_disabled
        self._agent: Agent = _create_agent()

    def run(
        self,
        eval_path: Path,
        baseline_path: Path,
    ) -> list[Phase3CaseResult]:
        """Execute the OpenAI Agents SDK AML agent on all eval cases.

        Parameters
        ----------
        eval_path : Path
            data/fixtures/eval.jsonl — 90 frozen EvalCase rows.
        baseline_path : Path
            artifacts/results.jsonl — 90 Phase 1 baseline CaseResult rows.

        Returns
        -------
        list[Phase3CaseResult]
            90 results in eval-file order, all tagged framework="openai_agents".

        Raises
        ------
        RuntimeError
            On missing case, SDK failure, or protocol invariant violation.
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

        run_config = RunConfig(tracing_disabled=self._tracing_disabled)
        results: list[Phase3CaseResult] = []

        for case in cases:
            baseline = baseline_map.get(case.case_id)
            if baseline is None:
                raise RuntimeError(
                    f"Case {case.case_id!r} absent from baseline results"
                )

            # PREPARE: evidence built before timer; agent already in self._agent.
            case_data = build_evidence(baseline)
            input_json = json.dumps(case_data)

            # EXECUTE: timer wraps only Runner.run_sync.
            t0 = time.perf_counter()
            try:
                sdk_result = _run_agent(self._agent, input_json, run_config)
            except Exception as exc:
                raise RuntimeError(
                    f"SDK execution failed for case {case.case_id!r}: {exc}"
                ) from exc
            latency_ms = (time.perf_counter() - t0) * 1000

            decision: OpenAIAgentDecision = sdk_result.final_output
            if not isinstance(decision, OpenAIAgentDecision):
                raise RuntimeError(
                    f"Case {case.case_id!r}: expected OpenAIAgentDecision, "
                    f"got {type(decision).__name__}"
                )

            agent_override = decision.disposition != baseline.disposition

            results.append(
                Phase3CaseResult(
                    framework=_FRAMEWORK,
                    case_id=case.case_id,
                    account_id=case.account_id,
                    disposition=decision.disposition,
                    decision_reason=decision.decision_reason,
                    sanctions_hits=baseline.sanctions_hits,
                    rule_firings=baseline.rule_firings,
                    anomaly_score=baseline.anomaly_score,
                    latency_ms=latency_ms,
                    agent_reasoning=decision.agent_reasoning,
                    agent_override=agent_override,
                    baseline_disposition=baseline.disposition,
                    human_review_flagged=decision.human_review_flagged,
                    tokens_used=0,
                    cost_usd=0.0,
                )
            )

        validate_phase3_results(results, cases, _FRAMEWORK)
        return results


# ── Internal helpers ──────────────────────────────────────────────────────────


def _create_agent() -> Agent:
    """Build the AML Triage Agent with DeterministicAMLModel.

    Called once in OpenAIAgentsRunner.__init__; the Agent is stateless and
    reusable across all 90 per-case Runner.run_sync calls.
    """
    model = DeterministicAMLModel()
    return Agent(
        name="AML Triage Agent",
        instructions=(
            "You are an AML triage specialist.\n"
            "Process the supplied case using the available tools:\n"
            "1. Call parse_evidence to extract and structure the case evidence.\n"
            "2. Call decide_disposition with the evidence to obtain a risk decision.\n"
            "3. Return the final structured result.\n"
            "Do not invent evidence. Do not alter the returned disposition. "
            "Do not call external services. Do not mention OFAC canonical names."
        ),
        model=model,
        tools=[parse_evidence, decide_disposition],
        output_type=OpenAIAgentDecision,
    )


def _run_agent(agent: Agent, input_json: str, run_config: RunConfig) -> Any:
    """Invoke Runner.run_sync for a single case.

    Thin wrapper around Runner.run_sync; used as the per-case timed operation.
    Agent and RunConfig are constructed outside this function.
    """
    return Runner.run_sync(
        agent,
        input=input_json,
        run_config=run_config,
        max_turns=10,
    )


def _build_case_data(baseline: CaseResult) -> dict[str, Any]:
    """Thin wrapper for backward compatibility; delegates to build_evidence."""
    return build_evidence(baseline)


def _load_eval_cases(path: Path) -> list[EvalCase]:
    """Thin wrapper for backward compatibility; delegates to load_eval_cases."""
    return load_eval_cases(path)


def _load_baseline_results(path: Path) -> dict[str, CaseResult]:
    """Thin wrapper for backward compatibility; delegates to load_baseline_results."""
    return load_baseline_results(path)


def _validate(results: list[Phase3CaseResult]) -> None:
    """Backward-compat shim: assert protocol invariants that don't require eval_cases.

    Tests that call _validate(results) directly continue to work.  The full
    validate_phase3_results() check (including eval-coverage invariants) runs
    inside OpenAIAgentsRunner.run() with proper eval_cases.
    """
    from collections import Counter

    if len(results) != EXPECTED_EVAL_SIZE:
        raise RuntimeError(
            f"OpenAIAgentsRunner produced {len(results)} results; "
            f"expected {EXPECTED_EVAL_SIZE}"
        )

    case_ids = [r.case_id for r in results]
    dup_counts = Counter(case_ids)
    dups = [cid for cid, n in dup_counts.items() if n > 1]
    if dups:
        raise RuntimeError(
            f"OpenAIAgentsRunner: duplicate case_ids: {dups}"
        )

    wrong = [r.case_id for r in results if r.framework != _FRAMEWORK]
    if wrong:
        raise RuntimeError(
            f"OpenAIAgentsRunner: {len(wrong)} results carry wrong framework tag "
            f"(expected {_FRAMEWORK!r}): {wrong[:5]}"
        )

    empty_fields = [r.case_id for r in results if not r.case_id or not r.account_id]
    if empty_fields:
        raise RuntimeError(
            f"OpenAIAgentsRunner: missing required fields in cases: {empty_fields}"
        )

    bad_tokens = [r.case_id for r in results if r.tokens_used != 0]
    if bad_tokens:
        raise RuntimeError(
            f"OpenAIAgentsRunner: non-zero tokens_used in cases: {bad_tokens}"
        )

    bad_cost = [r.case_id for r in results if r.cost_usd != 0.0]
    if bad_cost:
        raise RuntimeError(
            f"OpenAIAgentsRunner: non-zero cost_usd in cases: {bad_cost}"
        )
