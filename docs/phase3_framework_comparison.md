# Phase 3 Framework Comparison

This document describes the experimental design, implementation approach, measured
results, and interpretation of the Phase 3 cross-framework comparison.

---

## Experimental Question

Can three different agent-orchestration frameworks — LangGraph, CrewAI, and OpenAI
Agents SDK — produce equivalent AML case dispositions when given identical evidence,
an identical decision policy, and an identical evaluation set?

This is a **controlled equivalence test**, not a reasoning-quality benchmark.

---

## Controlled Variables

The following were held constant across all three frameworks:

| Variable | Value |
|---|---|
| Evaluation set | `data/fixtures/eval.jsonl` — 90 frozen `EvalCase` rows |
| Baseline evidence | `artifacts/results.jsonl` — 90 Phase 1 `CaseResult` rows (SanctionsHits, RuleFireings, AnomalyScore, baseline disposition) |
| Decision policy | `mock_llm_call` — 4-branch deterministic function; identical logic for all adapters |
| Output schema | `Phase3CaseResult` — Pydantic v2; all frameworks return the same structure |
| Metrics | `compute_framework_metrics` — same computation for all frameworks |
| Network access | None — all frameworks operate fully offline |
| Token usage | Zero — no LLM API is called |
| Cost | $0.00 |

---

## Independent Variable

The **orchestration framework only**.

Each framework's adapter wraps the shared `mock_llm_call` in its own execution model:
LangGraph uses typed state and conditional edge routing; CrewAI uses Agent/Task/Crew
abstraction; OpenAI Agents SDK uses Runner + function tools.

---

## Framework Implementations

### LangGraph (70 LOC)

LangGraph expresses the AML pipeline as a typed state graph:

- **State**: `TypedDict` carrying the case evidence, intermediate decisions, and
  framework results.
- **Routing**: `conditional_edges` on the `decision_node` output; deterministic
  branching based on `mock_llm_call` return values.
- **Graph semantics**: explicit node functions (`prepare_node`, `decision_node`,
  `output_node`) connected by directed edges.
- **Checkpoint suitability**: LangGraph's `MemorySaver` / `SqliteSaver` integrate
  naturally with the state type; human-in-the-loop pausing is straightforward to add.
- **Adapter**: `LangGraphRunner.run()` compiles the graph once at init, then calls
  `graph.invoke(initial_state)` per case.

LangGraph is the strongest fit for compliance workflows where routing logic must be
explicit, inspectable, and resumable at defined checkpoints.

---

### CrewAI (247 LOC)

CrewAI expresses the AML agent as an Agent/Task/Crew abstraction:

- **Custom LLM**: `DeterministicCrewAILLM` subclasses CrewAI's `BaseLLM` and routes
  calls to `mock_llm_call`; no external API is contacted.
- **Task context**: the evidence dict is passed as task context; the Agent's `goal`
  and `backstory` are set to AML investigator roles.
- **Execution mode**: sequential (single-agent crew with one task); hierarchical mode
  is avoided for compliance use because it introduces implicit state sharing.
- **Setup / runtime separation**: `Agent`, `Task`, and `Crew` are constructed before
  the per-case timer starts; only `crew.kickoff()` is timed.
- **No implicit long-term memory**: `memory=False` is set on the Agent to prevent
  cross-case state leakage.

CrewAI's role/task abstraction makes it intuitive for collaborative-agent prototypes,
but its implicit context passing makes it harder to audit than LangGraph for
compliance pipelines.

---

### OpenAI Agents SDK (376 LOC)

The OpenAI Agents SDK expresses the AML policy as a function-tool agent:

- **Custom model**: `DeterministicAMLModel` subclasses the SDK's `Model` interface
  and implements `get_response` to call `mock_llm_call`; no OpenAI API key is needed.
- **Function tools**: `analyse_case` is registered as a tool; the agent calls it to
  produce a structured `AMLDecision` response.
- **Execution**: `Runner.run_sync(agent, input_message, max_turns=2)` per case;
  tracing is disabled.
- **Structured output**: the `AMLDecision` schema is validated with Pydantic before
  being converted to `Phase3CaseResult`.
- **Validation shim**: `openai_agents_runner._validate()` provides an isolated
  function that mirrors the SDK's internal validation step, enabling targeted unit
  tests.

The OpenAI Agents SDK has the most concise surface area for tool-centric agents and
integrates naturally with OpenAI provider tooling, but requires more boilerplate to
inject a fully offline custom model.

---

## Comparison Dimensions

| Dimension | LangGraph | CrewAI | OpenAI Agents SDK |
|---|---|---|---|
| **Correctness parity** | Identical to other frameworks | Identical | Identical |
| **State visibility** | Explicit `TypedDict`; fully inspectable | Implicit task context | `RunResult` + tool call trace |
| **Routing determinism** | Explicit `conditional_edges` | Sequential; no branching needed | `max_turns` guard + tool dispatch |
| **Framework setup (one-time)** | Graph compile | Agent/Task/Crew construction | Agent + Model construction |
| **Testability** | High — graph nodes are plain functions | Medium — requires crew.kickoff() for full path | Medium — requires Runner for full path |
| **Offline model injection** | Straightforward (pass any callable) | Requires `BaseLLM` subclass | Requires `Model` subclass |
| **Human-in-the-loop** | Native checkpoint support | Callback hooks | Lifecycle hooks |
| **Checkpointing** | Native `MemorySaver` / `SqliteSaver` | Not native | Not native |
| **Error propagation** | Node exceptions surface cleanly | Exception in `kickoff()` | Exception in `run_sync()` |
| **Latency (p50, 90-case)** | 0.78 ms | 42.96 ms | 3.81 ms |
| **Latency (p95, 90-case)** | 0.98 ms | 63.27 ms | 4.17 ms |
| **Implementation LOC** | 70 | 247 | 376 |
| **Dependency footprint** | `langgraph` (in base extras) | `crewai` (compare extras) | `openai-agents` (compare extras) |

> Latency reflects framework orchestration overhead only. The underlying decision
> (`mock_llm_call`) is a deterministic function taking < 0.1 ms. CrewAI's higher
> latency comes from `crew.kickoff()` setup per case; LangGraph benefits from
> pre-compiled graph state.  Latency values are machine-specific.

---

## Results

All values read from `artifacts/phase3_comparison_metrics.json` (eval_size = 90,
generated 2026-07-12).

| Metric | LangGraph | CrewAI | OpenAI Agents SDK |
|---|---|---|---|
| Disposition accuracy | 78.89% | 78.89% | 78.89% |
| Weighted FCR | 17.22% | 17.22% | 17.22% |
| Override rate | 5.56% | 5.56% | 5.56% |
| Human-review rate | 16.67% | 16.67% | 16.67% |
| Latency p50 (ms) | 0.78 | 42.96 | 3.81 |
| Latency p95 (ms) | 0.98 | 63.27 | 4.17 |
| Implementation LOC | 70 | 247 | 376 |
| Tokens used | 0 | 0 | 0 |
| Cost | $0.00 | $0.00 | $0.00 |
| Dispositions agree | Yes | Yes | Yes |
| Reasoning agrees | Yes | Yes | Yes |
| Human-review flags agree | Yes | Yes | Yes |

---

## Key Finding

The three frameworks produced **identical policy outcomes** across all 90 eval cases
because the decision policy and evidence were held constant. Accuracy, weighted
false-clear rate, override rate, and human-review rate are equal by construction.

The meaningful differences are **architectural and operational**:

- **State visibility**: LangGraph exposes every intermediate decision; CrewAI and the
  OpenAI Agents SDK require more instrumentation to inspect sub-steps.
- **Routing explicitness**: LangGraph conditional edges are a first-class compile-time
  artifact; the others express routing through code control flow.
- **Checkpoint support**: LangGraph integrates persistency natively; the others require
  custom persistence layers.
- **Offline injection effort**: LangGraph accepts any callable; CrewAI and OpenAI
  Agents SDK require subclassing their model interface.
- **LOC**: LangGraph achieves the most direct mapping to the policy graph.

None of these differences affect prediction quality in Phase 3 because all frameworks
call the same policy function.

---

## Framework Decision Matrix

This matrix guides framework selection for future phases and similar projects.

| Use case | Recommended |
|---|---|
| Compliance workflows with explicit audit trails and checkpoints | **LangGraph** |
| Checkpoint-based human-in-the-loop review (SAR workflows) | **LangGraph** |
| Collaborative multi-role agent prototypes | **CrewAI** |
| Tool-centric agents with OpenAI provider ecosystem alignment | **OpenAI Agents SDK** |
| Minimal dependencies in production baseline | **LangGraph** (already in base install) |

**No framework is declared an absolute winner.** All three are viable for Phase 4
live-LLM evaluation. Framework choice in Phase 4 should be driven by:

- Production checkpoint and audit requirements → LangGraph
- Multi-agent task decomposition → CrewAI
- Provider API integration and tool schemas → OpenAI Agents SDK

---

## What Phase 4 Changes

Phase 4 replaces `mock_llm_call` with a real LLM provider call (OpenAI, Anthropic,
or similar). This is a **new experiment**, not a continuation of the Phase 3 result:

- In Phase 3, all frameworks call the same deterministic function with zero tokens.
  Agreement is expected by design.
- In Phase 4, each framework will call a live model. Agreement is no longer
  guaranteed — real reasoning introduces variability.
- Phase 4 will compare framework accuracy *against the frozen Phase 1 baseline*
  (`artifacts/metrics_baseline.json`), not against each other.
- The Phase 3 harness (eval set, evidence contract, shared metrics, CI workflows)
  is the control infrastructure that makes Phase 4 results interpretable.

The frozen `artifacts/phase3_comparison_metrics.json` is the baseline for any
Phase 4 comparison experiment.

---

## Claims Supported by This Project

The following claims are directly supported by measured artifacts in this repository.

**Allowed claims:**

- Built a deterministic AML triage baseline over approximately 5 million transactions
  and approximately 515,000 accounts (IBM AMLSim HI-Small).
- Created a frozen 90-case evaluation set covering five case types: IBM-labeled
  dispositions, sanctions hits, sanctions near-misses, rules-vs-anomaly conflicts,
  and typology coverage.
- Implemented equivalent orchestration in LangGraph, CrewAI, and OpenAI Agents SDK,
  each satisfying the `AMLAgentRunner` protocol.
- Achieved 90/90 framework disposition agreement under a shared deterministic policy.
- Improved controlled-policy accuracy from approximately 75.56% (Phase 1 baseline)
  to approximately 78.89% (Phase 2/3 shared policy).
- Reduced weighted false-clear rate from approximately 22.52% to approximately
  17.22% under the shared policy.
- Built offline, zero-token, zero-cost framework tests with 89% coverage and a
  CI gate at 85%.
- Added GitHub Actions CI workflows for the base suite and Phase 3 comparison suite.

**Disallowed claims (not supported by this project):**

- One framework produced more accurate results than another.
  *(All three call the same policy function; accuracy is equal by construction.)*
- A real LLM improved AML accuracy.
  *(Phase 3 uses a mock deterministic policy; no LLM is called.)*
- The system is production-ready for financial compliance.
  *(The system is a research baseline on synthetic data.)*
- The system replaces AML investigators or reduces the need for human review.
  *(Human-review routing is a feature, not an elimination of human judgment.)*
- Results generalize to real financial crime patterns.
  *(IBM AMLSim is a synthetic dataset; real transaction distributions differ.)*
- The agent discovered laundering patterns autonomously.
  *(All rules, thresholds, and anomaly features were engineered deterministically.)*
- The system achieves compliance with any regulator.
  *(AML typology rules are not a substitute for a full compliance program.)*
