"""
Phase 3 M2 — LangGraph adapter satisfying AMLAgentRunner.

Thin wrapper around the Phase 2 LangGraph evaluation pipeline.
All graph construction, node logic, and mock LLM policy are owned by
phase2_eval.run_langgraph_eval; no logic is duplicated here.

Latency methodology: inherited from run_langgraph_eval.run(), which wraps
graph.invoke() per case.  The compiled graph is built once before the loop.
See phase3_compare._shared for the full latency contract.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from aml_copilot.phase2_eval.run_langgraph_eval import (
    Phase2CaseResult,
    run as _p2_run,
)
from aml_copilot.phase3_compare._shared import (
    EXPECTED_EVAL_SIZE,
    load_eval_cases,
    validate_phase3_results,
)
from aml_copilot.schemas import Phase3CaseResult

_FRAMEWORK: str = "langgraph"


class LangGraphRunner:
    """
    AMLAgentRunner adapter for LangGraph (Phase 3 M2).

    Delegates entirely to the Phase 2 evaluation pipeline and injects
    framework="langgraph" into every result.  The graph topology, node
    functions, and mock LLM policy are unchanged.
    """

    framework_name: str = _FRAMEWORK

    def run(
        self,
        eval_path: Path,
        baseline_path: Path,
    ) -> list[Phase3CaseResult]:
        """
        Execute the LangGraph AML agent on all eval cases.

        Parameters
        ----------
        eval_path : Path
            data/fixtures/eval.jsonl — 90 frozen EvalCase rows.
        baseline_path : Path
            artifacts/results.jsonl — 90 Phase 1 baseline CaseResult rows.

        Returns
        -------
        list[Phase3CaseResult]
            90 results in eval-file order, all tagged framework="langgraph".

        Raises
        ------
        RuntimeError
            If result count, uniqueness, or framework-tag invariants fail.
        """
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as fh:
            tmp_path = Path(fh.name)
        try:
            p2_results: list[Phase2CaseResult] = _p2_run(
                eval_path=eval_path,
                baseline_path=baseline_path,
                out_path=tmp_path,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        results = [_to_phase3(r) for r in p2_results]
        eval_cases = load_eval_cases(eval_path)
        validate_phase3_results(results, eval_cases, _FRAMEWORK)
        return results


def _to_phase3(p2: Phase2CaseResult) -> Phase3CaseResult:
    """Inject framework tag into a Phase2CaseResult via Pydantic re-validation."""
    return Phase3CaseResult.model_validate({**p2.model_dump(), "framework": _FRAMEWORK})
