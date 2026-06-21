from __future__ import annotations

from aml_copilot.schemas import CaseResult, EvalCase, MetricsReport

# False-clear severity weights: a severity-3 FN costs 3× a severity-1 FN.
SEVERITY_WEIGHTS: dict[int, float] = {1: 1.0, 2: 2.0, 3: 3.0}


def compute_metrics(
    results: list[CaseResult],
    eval_cases: list[EvalCase],
) -> MetricsReport:
    """
    Compute all metrics from results + gold labels.
    Denominator guards: raises ValueError if ESCALATE gold count is 0.
    """
    ...


def save_metrics(report: MetricsReport, path: str) -> None:
    """Write MetricsReport to JSON at path and record its SHA-256 as the Phase 4 control row."""
    ...
