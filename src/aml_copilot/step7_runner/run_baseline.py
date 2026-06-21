"""
CLI entrypoint for the deterministic baseline pipeline.

Usage:
    python -m aml_copilot.step7_runner.run_baseline \\
        --eval data/fixtures/eval.jsonl \\
        --out  artifacts/results.jsonl
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    ...


def main() -> None:
    """
    Load all tools once at startup (index building happens once, not per case).
    For each EvalCase: run steps 2–5, call decision table, record CaseResult + latency_ms.
    Write results atomically (temp file + rename).
    Verify frozen artifact checksums before processing; abort on mismatch.
    """
    ...


if __name__ == "__main__":
    main()
