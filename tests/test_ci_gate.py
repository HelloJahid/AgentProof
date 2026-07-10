"""Phase 6 closer: the eval suite as a merge blocker, end to end."""

import json
from pathlib import Path

from agentproof.evals import CaseResult, CheckResult, SuiteResult
from agentproof.evals.ci_gate import run_gate
from demo.researcher.evalsuite import main


def make_suite(passed: bool) -> SuiteResult:
    return SuiteResult(
        results=[
            CaseResult(
                case_id="only-case",
                outcome="finished",
                path=["model"],
                final_answer="x",
                checks=[
                    CheckResult(check="c", dimension="quality", passed=passed, reason="because")
                ],
                passed=passed,
                trace_path="t.jsonl",
            )
        ]
    )


def test_gate_exit_codes_speak_ci(tmp_path: Path, capsys) -> None:
    assert run_gate(make_suite(passed=True)) == 0
    assert run_gate(make_suite(passed=False)) == 1
    out = capsys.readouterr().out
    assert "eval gate: PASS" in out
    assert "eval gate: FAIL -- broken cases: only-case" in out


def test_gate_saves_the_scorecard_as_a_json_artifact(tmp_path: Path) -> None:
    scorecard = tmp_path / "artifacts" / "scorecard.json"

    run_gate(make_suite(passed=True), scorecard_path=scorecard)

    saved = json.loads(scorecard.read_text(encoding="utf-8"))
    assert saved["results"][0]["case_id"] == "only-case"


def test_the_real_golden_suite_passes_exactly_as_ci_will_run_it(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "--trace-dir",
            str(tmp_path / "traces"),
            "--scorecard",
            str(tmp_path / "scorecard.json"),
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[PASS] gtc-announcement" in out
    assert "[PASS] search-down" in out
    assert "[PASS] greeting-no-search" in out
    # Every golden verdict left re-verifiable evidence:
    assert (tmp_path / "traces" / "gtc-announcement.trace.jsonl").exists()
    assert (tmp_path / "scorecard.json").exists()


def test_a_broken_expectation_turns_the_gate_red(tmp_path: Path) -> None:
    # Same golden scripts, but the dataset now demands the impossible --
    # exactly what a regression looks like from the gate's point of view.
    dataset = tmp_path / "strict.jsonl"
    dataset.write_text(
        '{"id": "gtc-announcement", "query": "What did NVIDIA announce at GTC 2025?", '
        '"answer_pattern": "THIS_WILL_NEVER_APPEAR"}\n',
        encoding="utf-8",
    )

    exit_code = main(["--dataset", str(dataset), "--trace-dir", str(tmp_path / "traces")])

    assert exit_code == 1
