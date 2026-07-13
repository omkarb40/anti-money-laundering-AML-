"""
AMLAgentRunner Protocol — contract for Phase 3 framework adapters.

IMPORTANT — runtime_checkable limitation
-----------------------------------------
isinstance(obj, AMLAgentRunner) performs structural checking only: it verifies
that the object exposes a 'framework_name' attribute and a 'run' method.
It does NOT validate:
  - return type       (must be list[Phase3CaseResult])
  - result count      (must equal the number of eval cases)
  - unique case IDs   (no duplicates permitted)
  - framework tag     (every result.framework must equal self.framework_name)

Callers (the M5 comparison runner) must validate all four invariants explicitly
after calling run().
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from aml_copilot.schemas import Phase3CaseResult

PROTOCOL_VERSION: str = "1.0"


@runtime_checkable
class AMLAgentRunner(Protocol):
    """
    Structural protocol satisfied by every Phase 3 framework adapter.

    Attributes
    ----------
    framework_name : str
        Short identifier for the framework.
        Expected values: "langgraph", "crewai", "openai_agents".

    Methods
    -------
    run(eval_path, baseline_path) -> list[Phase3CaseResult]
        Execute the AML decision agent on all eval cases.

        Parameters
        ----------
        eval_path : Path
            data/fixtures/eval.jsonl — 90 frozen EvalCase rows.
        baseline_path : Path
            artifacts/results.jsonl — 90 Phase 1 CaseResult rows.
            Provides pre-computed tool outputs; runners need no raw
            transaction data.

        Returns
        -------
        list[Phase3CaseResult]
            One result per eval case, in the same order as the input file.
            Invariants the caller must verify:
              - len(results) == len(eval_cases)
              - all r.framework == self.framework_name
              - no duplicate case_ids
              - all r.cost_usd == 0.0 and r.tokens_used == 0 in mock mode
    """

    framework_name: str

    def run(
        self,
        eval_path: Path,
        baseline_path: Path,
    ) -> list[Phase3CaseResult]:
        ...
