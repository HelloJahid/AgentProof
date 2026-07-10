"""Phase 6: checks that survive a live, nondeterministic model."""

from agentproof.evals import (
    EvalCase,
    PathIncludesSubsequence,
    UsedExpectedTools,
    WithinBudgets,
)
from agentproof.state import AgentState, ToolResult
from agentproof.trace.replay import ReplayStep, RunReplay


def make_replay(path: list[str], final_state: AgentState | None = None) -> RunReplay:
    steps = []
    for index, name in enumerate(path):
        is_last = index == len(path) - 1
        state = final_state if (is_last and final_state is not None) else AgentState(query="q")
        steps.append(ReplayStep(name=name, duration_ms=1.0, state=state))
    return RunReplay(
        run_id="r1",
        query="q",
        instructions="",
        steps=steps,
        outcome="finished",
        final_answer="answer",
    )


LIVE_PATH = ["prepare", "model", "tools", "model", "tools", "model"]  # searched twice


def test_subsequence_check_tolerates_extra_live_steps() -> None:
    case = EvalCase(id="c", query="q", expected_path=["model", "tools", "model"])

    result = PathIncludesSubsequence().evaluate(case, make_replay(LIVE_PATH))

    assert result.passed  # exact match would have failed this healthy run


def test_subsequence_check_still_catches_a_missing_phase() -> None:
    case = EvalCase(id="c", query="q", expected_path=["model", "tools", "model"])

    # The agent answered without ever running tools:
    result = PathIncludesSubsequence().evaluate(case, make_replay(["prepare", "model"]))

    assert not result.passed
    assert "missing" in result.reason


def test_subsequence_requires_order_not_just_presence() -> None:
    case = EvalCase(id="c", query="q", expected_path=["tools", "model"])

    # 'tools' and 'model' both appear, but never tools-then-model:
    result = PathIncludesSubsequence().evaluate(case, make_replay(["model", "tools"]))

    assert not result.passed


def test_used_expected_tools_reads_the_final_state() -> None:
    case = EvalCase(id="c", query="q", expected_tools=["web_search"])
    final = AgentState(query="q")
    final.tool_results.append(ToolResult(call_id="c1", name="web_search", output="{}"))

    ok = UsedExpectedTools().evaluate(case, make_replay(LIVE_PATH, final))
    missing = UsedExpectedTools().evaluate(case, make_replay(["prepare", "model"]))

    assert ok.passed
    assert not missing.passed
    assert "web_search" in missing.reason


def test_budgets_pass_within_and_fail_beyond() -> None:
    case = EvalCase(id="c", query="q", max_steps=6, max_total_tokens=1000)
    final = AgentState(query="q")
    final.input_tokens, final.output_tokens = 700, 200

    ok = WithinBudgets().evaluate(case, make_replay(LIVE_PATH, final))
    assert ok.passed and ok.dimension == "system"
    assert "900 tokens" in ok.reason

    greedy = AgentState(query="q")
    greedy.input_tokens, greedy.output_tokens = 900, 200
    over = WithinBudgets().evaluate(
        EvalCase(id="c", query="q", max_steps=3, max_total_tokens=1000),
        make_replay(LIVE_PATH, greedy),
    )
    assert not over.passed
    assert "6 steps > budget 3" in over.reason
    assert "1100 tokens > budget 1000" in over.reason


def test_all_three_report_not_applicable_without_references() -> None:
    bare = EvalCase(id="c", query="q")
    replay = make_replay(LIVE_PATH)

    for check in (PathIncludesSubsequence(), UsedExpectedTools(), WithinBudgets()):
        result = check.evaluate(bare, replay)
        assert not result.applicable and result.passed
