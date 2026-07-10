"""Phase 4: datasets load strictly; rule evaluators judge replays with reasons."""

from pathlib import Path

import pytest

from agentproof.errors import EvalFailure
from agentproof.evals import (
    AnswerMatchesReference,
    EvalCase,
    RunSucceeded,
    TookExpectedPath,
    load_dataset,
)
from agentproof.trace.replay import RunReplay


def make_replay(**overrides) -> RunReplay:
    defaults = dict(
        run_id="r1",
        query="Umbrella today in Melbourne?",
        instructions="",
        steps=[],
        outcome="finished",
        final_answer="18.0 degrees, bring an umbrella.",
    )
    return RunReplay.model_validate({**defaults, **overrides})


# --- datasets ----------------------------------------------------------------


def test_dataset_loads_valid_cases(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id": "weather-1", "query": "Umbrella today?", '
        '"answer_pattern": "umbrella", "expected_path": ["model", "tools", "model"]}\n'
        '{"id": "weather-2", "query": "Temperature?"}\n',
        encoding="utf-8",
    )

    cases = load_dataset(path)

    assert [case.id for case in cases] == ["weather-1", "weather-2"]
    assert cases[0].expected_path == ["model", "tools", "model"]
    assert cases[1].reference_answer is None  # references are optional


def test_malformed_and_duplicate_datasets_fail_loudly(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x"}\n', encoding="utf-8")  # missing required query
    with pytest.raises(EvalFailure, match="line 1"):
        load_dataset(bad)

    dupes = tmp_path / "dupes.jsonl"
    dupes.write_text(
        '{"id": "same", "query": "a"}\n{"id": "same", "query": "b"}\n', encoding="utf-8"
    )
    with pytest.raises(EvalFailure, match="duplicate"):
        load_dataset(dupes)


# --- rule evaluators -----------------------------------------------------------


def test_run_succeeded_passes_finished_and_fails_crashed_runs() -> None:
    case = EvalCase(id="c", query="q")

    ok = RunSucceeded().evaluate(case, make_replay())
    assert ok.passed and ok.dimension == "task_completion"

    crashed = RunSucceeded().evaluate(
        case, make_replay(outcome="failed", final_answer=None, error_message="boom")
    )
    assert not crashed.passed
    assert "boom" in crashed.reason  # the reason carries the evidence


def test_answer_check_supports_exact_and_pattern_references() -> None:
    exact_case = EvalCase(id="c", query="q", reference_answer="18.0 degrees, bring an umbrella.")
    assert AnswerMatchesReference().evaluate(exact_case, make_replay()).passed

    pattern_case = EvalCase(id="c", query="q", answer_pattern=r"umbrella")
    assert AnswerMatchesReference().evaluate(pattern_case, make_replay()).passed

    wrong = AnswerMatchesReference().evaluate(
        pattern_case, make_replay(final_answer="clear skies all day")
    )
    assert not wrong.passed
    assert "umbrella" in wrong.reason  # says what was expected


def test_checks_report_not_applicable_instead_of_guessing() -> None:
    bare_case = EvalCase(id="c", query="q")  # no references at all

    answer = AnswerMatchesReference().evaluate(bare_case, make_replay())
    path = TookExpectedPath().evaluate(bare_case, make_replay())

    assert not answer.applicable and answer.passed
    assert not path.applicable and path.passed


def test_trajectory_check_compares_the_actual_path() -> None:
    case = EvalCase(id="c", query="q", expected_path=["model", "tools", "model"])

    replay = make_replay()
    # Build a replay whose steps produce the right path:
    from agentproof.state import AgentState
    from agentproof.trace.replay import ReplayStep

    def step(name: str) -> ReplayStep:
        return ReplayStep(name=name, duration_ms=1.0, state=AgentState(query="q"))

    good = replay.model_copy(update={"steps": [step("model"), step("tools"), step("model")]})
    bad = replay.model_copy(update={"steps": [step("model")]})

    assert TookExpectedPath().evaluate(case, good).passed
    result = TookExpectedPath().evaluate(case, bad)
    assert not result.passed
    assert "['model']" in result.reason  # shows the path actually taken
