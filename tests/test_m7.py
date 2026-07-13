"""
M7 release-readiness tests.

Verifies that documentation, artifacts, and safeguards meet the Phase 3
release criteria.  Tests are intentionally non-brittle: they check critical
invariants and claims, not exact paragraph wording.

All tests are offline — no raw data, no API keys, no network.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# ── Root / doc paths ──────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent
_README = _ROOT / "README.md"
_ARCH = _ROOT / "docs" / "architecture.md"
_REPRO = _ROOT / "docs" / "reproducibility.md"
_P3_DOC = _ROOT / "docs" / "phase3_framework_comparison.md"
_MMD = _ROOT / "docs" / "images" / "phase3_architecture.mmd"
_ARTIFACT = _ROOT / "artifacts" / "phase3_comparison_metrics.json"
_CHECKSUMS = _ROOT / "artifacts" / "checksums.sha256"
_TEST_YML = _ROOT / ".github" / "workflows" / "test.yml"
_COMPARE_YML = _ROOT / ".github" / "workflows" / "phase3-compare.yml"
_MINI_EVAL = _ROOT / "tests" / "fixtures" / "phase3_mini_eval.jsonl"
_MINI_BASELINE = _ROOT / "tests" / "fixtures" / "phase3_mini_baseline.jsonl"
_CONTRIBUTING = _ROOT / "CONTRIBUTING.md"

pytestmark = pytest.mark.compare


# ── README content invariants ─────────────────────────────────────────────────

class TestReadmeContent:
    def test_readme_exists(self):
        assert _README.exists(), "README.md is missing"

    def test_readme_contains_phase23_accuracy(self):
        text = _README.read_text()
        # Phase 2/3 accuracy is 78.89% — must appear in some recognisable form
        assert "78.89" in text or "0.7889" in text, (
            "README does not contain Phase 2/3 accuracy (78.89%)"
        )

    def test_readme_contains_phase1_accuracy(self):
        text = _README.read_text()
        assert "75.56" in text or "0.7556" in text, (
            "README does not contain Phase 1 accuracy (75.56%)"
        )

    def test_readme_states_future_llm_work_not_done(self):
        text = _README.read_text()
        # README must indicate live LLM integration is future/planned work.
        assert (
            "Future Work" in text
            or "future work" in text
            or "Not started" in text
            or "not started" in text
            or "Planned" in text
        ), (
            "README does not indicate that live LLM integration is future/planned work"
        )

    def test_readme_does_not_claim_live_llm_in_phases_1_3(self):
        text = _README.read_text()
        # The phrase "no live model calls" or "no LLM API" should appear somewhere
        # to acknowledge the limitation. Also check for absence of problematic phrases.
        bad_phrases = [
            "real LLM improves",
            "live model improves",
            "LLM discovers",
            "production-ready compliance",
        ]
        for phrase in bad_phrases:
            assert phrase.lower() not in text.lower(), (
                f"README contains unsupported claim: '{phrase}'"
            )

    def test_readme_quick_start_mini_paths_exist(self):
        assert _MINI_EVAL.exists(), (
            "Quick-start mini eval path tests/fixtures/phase3_mini_eval.jsonl not found"
        )
        assert _MINI_BASELINE.exists(), (
            "Quick-start mini baseline tests/fixtures/phase3_mini_baseline.jsonl not found"
        )

    def test_readme_references_phase3_doc(self):
        text = _README.read_text()
        assert "phase3_framework_comparison" in text, (
            "README does not reference docs/phase3_framework_comparison.md"
        )

    def test_readme_references_architecture_doc(self):
        text = _README.read_text()
        assert "architecture.md" in text, (
            "README does not reference docs/architecture.md"
        )

    def test_readme_references_reproducibility_doc(self):
        text = _README.read_text()
        assert "reproducibility.md" in text, (
            "README does not reference docs/reproducibility.md"
        )

    def test_readme_no_absolute_user_paths(self):
        text = _README.read_text()
        assert "/Users/" not in text, "README contains an absolute /Users/ path"
        assert "/home/" not in text, "README contains an absolute /home/ path"

    def test_readme_no_framework_superiority_claim(self):
        text = _README.read_text()
        bad_phrases = [
            "LangGraph is more accurate",
            "CrewAI is more accurate",
            "OpenAI Agents is more accurate",
            "one framework outperforms",
        ]
        for phrase in bad_phrases:
            assert phrase.lower() not in text.lower(), (
                f"README contains framework superiority claim: '{phrase}'"
            )

    def test_readme_limitations_section_present(self):
        text = _README.read_text()
        assert "## Limitations" in text or "## limitations" in text.lower(), (
            "README does not have a Limitations section"
        )

    def test_readme_data_attribution_present(self):
        text = _README.read_text()
        assert "AMLSim" in text and "OFAC" in text, (
            "README missing data attribution for AMLSim or OFAC"
        )


# ── Required documentation files ─────────────────────────────────────────────

class TestDocumentationFiles:
    def test_phase3_comparison_doc_exists(self):
        assert _P3_DOC.exists(), "docs/phase3_framework_comparison.md is missing"

    def test_architecture_doc_exists(self):
        assert _ARCH.exists(), "docs/architecture.md is missing"

    def test_reproducibility_doc_exists(self):
        assert _REPRO.exists(), "docs/reproducibility.md is missing"

    def test_mermaid_diagram_source_exists(self):
        assert _MMD.exists(), "docs/images/phase3_architecture.mmd is missing"

    def test_contributing_guide_exists(self):
        assert _CONTRIBUTING.exists(), "CONTRIBUTING.md is missing"

    def test_phase3_doc_has_experimental_question(self):
        text = _P3_DOC.read_text()
        assert "Experimental Question" in text or "experimental question" in text.lower()

    def test_phase3_doc_has_claims_section(self):
        text = _P3_DOC.read_text()
        assert "Claims Supported" in text, (
            "phase3_framework_comparison.md missing 'Claims Supported' section"
        )

    def test_phase3_doc_has_disallowed_claims(self):
        text = _P3_DOC.read_text()
        assert "Disallowed" in text or "disallowed" in text.lower(), (
            "phase3_framework_comparison.md missing 'Disallowed claims' section"
        )

    def test_phase3_doc_states_identical_accuracy(self):
        text = _P3_DOC.read_text()
        assert "78.89" in text or "0.7889" in text, (
            "phase3_framework_comparison.md does not report Phase 3 accuracy"
        )

    def test_mermaid_file_has_mermaid_content(self):
        text = _MMD.read_text()
        assert "flowchart" in text or "graph" in text, (
            "phase3_architecture.mmd does not look like a Mermaid diagram"
        )


# ── No stale references ───────────────────────────────────────────────────────

class TestNoStaleReferences:
    def test_no_isolation_forest_in_source(self):
        src_dir = _ROOT / "src"
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text(errors="replace")
            assert "IsolationForest" not in content, (
                f"Stale IsolationForest reference in {py_file}"
            )

    def test_no_isolation_forest_in_docs(self):
        docs_dir = _ROOT / "docs"
        for md_file in docs_dir.rglob("*.md"):
            content = md_file.read_text(errors="replace")
            assert "IsolationForest" not in content, (
                f"Stale IsolationForest reference in {md_file}"
            )

    def test_no_absolute_users_paths_in_docs(self):
        docs_dir = _ROOT / "docs"
        for md_file in docs_dir.rglob("*.md"):
            content = md_file.read_text(errors="replace")
            assert "/Users/" not in content, (
                f"Absolute /Users/ path found in {md_file}"
            )

    def test_no_absolute_users_paths_in_workflows(self):
        workflows = _ROOT / ".github" / "workflows"
        for yml_file in workflows.rglob("*.yml"):
            content = yml_file.read_text(errors="replace")
            assert "/Users/" not in content, (
                f"Absolute /Users/ path found in {yml_file}"
            )

    def test_readme_no_stale_phase_roadmap(self):
        text = _README.read_text()
        # README must distinguish what is done from what is future/planned.
        # Accepts either the old phase-table format ("Phase 4" + "Not started"/"Planned")
        # or the new Future Work section format introduced in the README rewrite.
        phase_table_format = "Phase 4" in text and (
            "Not started" in text or "Planned" in text
        )
        future_work_format = "Future Work" in text or "future work" in text
        assert phase_table_format or future_work_format, (
            "README does not distinguish completed work from future plans"
        )


# ── Canonical eval path safeguard ─────────────────────────────────────────────

class TestCanonicalEvalSafeguard:
    def test_is_canonical_eval_path_true_for_canonical(self):
        from aml_copilot.phase3_compare.run_comparison import _is_canonical_eval_path
        canonical = _ROOT / "data" / "fixtures" / "eval.jsonl"
        assert _is_canonical_eval_path(canonical) is True

    def test_is_canonical_eval_path_false_for_mini(self):
        from aml_copilot.phase3_compare.run_comparison import _is_canonical_eval_path
        assert _is_canonical_eval_path(_MINI_EVAL) is False

    def test_is_canonical_eval_path_false_for_tmp(self, tmp_path):
        from aml_copilot.phase3_compare.run_comparison import _is_canonical_eval_path
        fake = tmp_path / "eval.jsonl"
        fake.write_text("")
        assert _is_canonical_eval_path(fake) is False

    def test_validate_eval_mode_passes_with_90_canonical_cases(self):
        from aml_copilot.phase3_compare.run_comparison import _validate_eval_mode
        from aml_copilot.schemas import EvalCase

        canonical = _ROOT / "data" / "fixtures" / "eval.jsonl"
        if not canonical.exists():
            pytest.skip("Canonical eval.jsonl not present; skipping guard test")

        cases = []
        with open(canonical) as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(EvalCase.model_validate_json(line))
        _validate_eval_mode(canonical, cases)  # must not raise

    def test_validate_eval_mode_fails_with_89_canonical_cases(self, tmp_path):
        from aml_copilot.phase3_compare.run_comparison import (
            _is_canonical_eval_path, _validate_eval_mode, _CANONICAL_EVAL,
        )
        from aml_copilot.schemas import EvalCase

        # Monkeypatch: we can't make a tmp path resolve to the canonical path,
        # so we test _validate_eval_mode by calling it with the canonical path
        # but a list of 89 items.
        canonical = _ROOT / "data" / "fixtures" / "eval.jsonl"
        if not canonical.exists():
            pytest.skip("Canonical eval.jsonl not present; skipping guard test")

        cases = []
        with open(canonical) as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(EvalCase.model_validate_json(line))
        cases_89 = cases[:89]

        with pytest.raises(RuntimeError, match="90 cases"):
            _validate_eval_mode(canonical, cases_89)

    def test_validate_eval_mode_passes_for_mini_5_cases(self):
        from aml_copilot.phase3_compare.run_comparison import _validate_eval_mode
        from aml_copilot.schemas import EvalCase

        cases = []
        with open(_MINI_EVAL) as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(EvalCase.model_validate_json(line))
        assert len(cases) == 5
        _validate_eval_mode(_MINI_EVAL, cases)  # must not raise

    def test_validate_eval_mode_passes_for_arbitrary_tmp_fixture(self, tmp_path):
        from aml_copilot.phase3_compare.run_comparison import _validate_eval_mode
        from aml_copilot.schemas import EvalCase

        # Load mini cases and use them with an arbitrary tmp path — no size constraint
        cases = []
        with open(_MINI_EVAL) as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(EvalCase.model_validate_json(line))
        arbitrary = tmp_path / "my_eval.jsonl"
        arbitrary.write_text("placeholder")
        _validate_eval_mode(arbitrary, cases)  # must not raise


# ── Official comparison artifact ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def _m7_artifact():
    if not _ARTIFACT.exists():
        pytest.skip("artifacts/phase3_comparison_metrics.json not found")
    return json.loads(_ARTIFACT.read_text())


@pytest.fixture(scope="module")
def _m7_artifact_schema(_m7_artifact):
    from aml_copilot.schemas import Phase3ComparisonMetrics
    return Phase3ComparisonMetrics.model_validate(_m7_artifact)


class TestComparisonArtifact:
    def test_artifact_schema_validates(self, _m7_artifact_schema):
        assert _m7_artifact_schema is not None

    def test_artifact_eval_size_is_90(self, _m7_artifact_schema):
        assert _m7_artifact_schema.eval_size == 90

    def test_artifact_comparison_passed(self, _m7_artifact_schema):
        assert _m7_artifact_schema.comparison_passed is True

    def test_artifact_exactly_three_frameworks(self, _m7_artifact_schema):
        assert len(_m7_artifact_schema.frameworks) == 3

    def test_artifact_framework_names(self, _m7_artifact_schema):
        names = {m.framework for m in _m7_artifact_schema.frameworks}
        assert names == {"langgraph", "crewai", "openai_agents"}

    def test_artifact_all_dispositions_agree(self, _m7_artifact_schema):
        assert _m7_artifact_schema.all_dispositions_agree is True

    def test_artifact_all_reasoning_agree(self, _m7_artifact_schema):
        assert _m7_artifact_schema.all_reasoning_agree is True

    def test_artifact_all_human_review_flags_agree(self, _m7_artifact_schema):
        assert _m7_artifact_schema.all_human_review_flags_agree is True

    def test_artifact_all_costs_zero(self, _m7_artifact_schema):
        assert _m7_artifact_schema.all_costs_zero is True

    def test_artifact_all_tokens_zero(self, _m7_artifact_schema):
        assert _m7_artifact_schema.all_tokens_zero is True

    def test_artifact_framework_accuracy_matches_readme(self, _m7_artifact_schema):
        for m in _m7_artifact_schema.frameworks:
            assert abs(m.disposition_accuracy - 0.7888888889) < 0.001, (
                f"{m.framework} accuracy {m.disposition_accuracy} deviates from expected 0.7889"
            )

    def test_artifact_generated_at_is_timezone_aware(self, _m7_artifact_schema):
        assert _m7_artifact_schema.generated_at.tzinfo is not None, (
            "generated_at is timezone-naive"
        )

    def test_artifact_framework_order_matches_registry(self, _m7_artifact_schema):
        expected_order = ["langgraph", "crewai", "openai_agents"]
        actual_order = [m.framework for m in _m7_artifact_schema.frameworks]
        assert actual_order == expected_order, (
            f"Framework order {actual_order} != registry order {expected_order}"
        )


# ── Checksums integrity ────────────────────────────────────────────────────────

class TestChecksumsIntegrity:
    def test_checksums_file_exists(self):
        assert _CHECKSUMS.exists(), "artifacts/checksums.sha256 not found"

    def test_no_phase3_artifact_in_checksums(self):
        text = _CHECKSUMS.read_text()
        assert "phase3_comparison" not in text, (
            "phase3_comparison_metrics.json must not appear in checksums.sha256"
        )

    def test_no_phase2_langgraph_metrics_in_checksums(self):
        text = _CHECKSUMS.read_text()
        assert "phase2_langgraph_metrics" not in text, (
            "phase2_langgraph_metrics.json must not appear in checksums.sha256"
        )

    def test_checksums_contains_expected_frozen_files(self):
        text = _CHECKSUMS.read_text()
        expected = [
            "ground_truth_matches.csv",
            "thresholds.py",
            "eval.jsonl",
            "metrics_baseline.json",
        ]
        for name in expected:
            assert name in text, f"Frozen artifact '{name}' missing from checksums.sha256"


# ── Documentation link audit ──────────────────────────────────────────────────

class TestDocumentationLinks:
    def _check_relative_links(self, doc_path: Path):
        """Verify that relative Markdown links in doc_path point to existing files."""
        import re
        text = doc_path.read_text()
        # Find Markdown links: [text](path) — skip URLs and anchors
        links = re.findall(r'\[.*?\]\(([^)]+)\)', text)
        doc_dir = doc_path.parent
        broken = []
        for link in links:
            if link.startswith("http") or link.startswith("#"):
                continue
            # Strip anchors from file paths
            path_part = link.split("#")[0]
            if not path_part:
                continue
            target = (doc_dir / path_part).resolve()
            if not target.exists():
                broken.append(link)
        return broken

    def test_readme_relative_links(self):
        broken = self._check_relative_links(_README)
        assert not broken, f"README has broken relative links: {broken}"

    def test_architecture_relative_links(self):
        broken = self._check_relative_links(_ARCH)
        assert not broken, f"docs/architecture.md has broken relative links: {broken}"

    def test_reproducibility_relative_links(self):
        broken = self._check_relative_links(_REPRO)
        assert not broken, f"docs/reproducibility.md has broken relative links: {broken}"

    def test_phase3_doc_relative_links(self):
        broken = self._check_relative_links(_P3_DOC)
        assert not broken, (
            f"docs/phase3_framework_comparison.md has broken relative links: {broken}"
        )


# ── YAML workflow validation ──────────────────────────────────────────────────

class TestCIWorkflows:
    def test_test_yml_parses(self):
        yaml = pytest.importorskip("yaml")
        content = _TEST_YML.read_text()
        parsed = yaml.safe_load(content)
        assert parsed is not None
        assert "jobs" in parsed

    def test_compare_yml_parses(self):
        yaml = pytest.importorskip("yaml")
        content = _COMPARE_YML.read_text()
        parsed = yaml.safe_load(content)
        assert parsed is not None
        assert "jobs" in parsed

    def test_test_yml_uses_correct_pytest_expression(self):
        text = _TEST_YML.read_text()
        assert "not integration" in text and "not live" in text, (
            "test.yml does not skip integration and live tests"
        )

    def test_compare_yml_has_coverage_gate(self):
        text = _COMPARE_YML.read_text()
        assert "cov-fail-under=85" in text or "--cov-fail-under 85" in text, (
            "phase3-compare.yml missing 85% coverage gate"
        )

    def test_compare_yml_validates_artifact(self):
        text = _COMPARE_YML.read_text()
        assert "model_validate" in text or "Phase3ComparisonMetrics" in text, (
            "phase3-compare.yml does not validate the comparison artifact with Pydantic"
        )

    def test_test_yml_no_secret_requirements(self):
        text = _TEST_YML.read_text()
        assert "OPENAI_API_KEY" not in text
        assert "ANTHROPIC_API_KEY" not in text

    def test_compare_yml_no_secret_requirements(self):
        text = _COMPARE_YML.read_text()
        assert "OPENAI_API_KEY" not in text
        assert "ANTHROPIC_API_KEY" not in text


# ── JSON/JSONL trailing-newline check ─────────────────────────────────────────

class TestTrailingNewlines:
    def _check_newline(self, path: Path) -> bool:
        raw = path.read_bytes()
        return raw.endswith(b"\n")

    def test_phase3_comparison_metrics_trailing_newline(self):
        if not _ARTIFACT.exists():
            pytest.skip("artifacts/phase3_comparison_metrics.json not present")
        assert self._check_newline(_ARTIFACT), (
            "artifacts/phase3_comparison_metrics.json missing trailing newline"
        )

    def test_mini_eval_trailing_newline(self):
        assert self._check_newline(_MINI_EVAL), (
            "tests/fixtures/phase3_mini_eval.jsonl missing trailing newline"
        )

    def test_mini_baseline_trailing_newline(self):
        assert self._check_newline(_MINI_BASELINE), (
            "tests/fixtures/phase3_mini_baseline.jsonl missing trailing newline"
        )

    def test_expected_comparison_trailing_newline(self):
        expected = _ROOT / "tests" / "fixtures" / "phase3_expected_comparison.json"
        assert self._check_newline(expected), (
            "tests/fixtures/phase3_expected_comparison.json missing trailing newline"
        )


# ── CLI help exits 0 ──────────────────────────────────────────────────────────

class TestCLIHelp:
    def test_run_baseline_help_exits_0(self):
        import subprocess
        result = subprocess.run(
            ["python", "-m", "aml_copilot.step7_runner.run_baseline", "--help"],
            cwd=str(_ROOT),
            capture_output=True,
        )
        assert result.returncode == 0, (
            f"run_baseline --help exited {result.returncode}: {result.stderr.decode()}"
        )

    def test_run_comparison_help_exits_0(self):
        import subprocess
        result = subprocess.run(
            ["python", "-m", "aml_copilot.phase3_compare.run_comparison", "--help"],
            cwd=str(_ROOT),
            capture_output=True,
        )
        assert result.returncode == 0, (
            f"run_comparison --help exited {result.returncode}: {result.stderr.decode()}"
        )
