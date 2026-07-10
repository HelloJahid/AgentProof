"""Phase 4 closer: dataset in, judged scorecard out -- fully mocked."""

from pathlib import Path

from pydantic import BaseModel, Field

from agentproof.evals import (
    AnswerMatchesReference,
    EvalCase,
    RunSucceeded,
    TookExpectedPath,
    render_scorecard,
    run_suite,
)
from agentproof.machine import StateMachine
from agentproof.state import AgentState, ToolCall
from agentproof.steps import ToolExecStep, react_router
from agentproof.tools import MockTransport, Tool, ToolExecutor, ToolRegistry

EVALUATORS = [RunSucceeded(), AnswerMatchesReference(), TookExpectedPath()]


class WeatherArgs(BaseModel):
    location: str = Field(min_length=2)


class WeatherReport(BaseModel):
    temperature: float
    rain_probability: int = Field(ge=0, le=100)


class ScriptedModelStep:
    """Asks for the weather, then answers -- or degrades politely on error."""

    name = "model"

    def run(self, state: AgentState) -> AgentState:
        if not state.tool_results:
            state.pending_tool_calls.append(
                ToolCall(id="c1", name="get_weather", arguments={"location": "Melbourne"})
            )
            return state
        result = state.tool_results[-1]
        if result.is_error:
            state.final_answer = "Sorry, the weather service is unavailable."
            return state
        report = WeatherReport.model_validate_json(result.output)
        advice = "bring an umbrella" if report.rain_probability > 50 else "no umbrella needed"
        state.final_answer = f"{report.temperature} degrees, {advice}."
        return state


def weather_agent_factory(script: dict[str, list]) -> "StateMachine":
    registry = ToolRegistry(
        [
            Tool(
                name="get_weather",
                description="Current weather.",
                input_model=WeatherArgs,
                output_model=WeatherReport,
            )
        ]
    )
    executor = ToolExecutor(registry, MockTransport(script), max_attempts=2)
    return StateMachine(
        [ScriptedModelStep(), ToolExecStep(executor)], router=react_router, start="model"
    )


CASES = [
    EvalCase(
        id="rainy-day",
        query="Umbrella today?",
        answer_pattern=r"umbrella",
        expected_path=["model", "tools", "model"],
    ),
    EvalCase(
        id="api-down",
        query="Umbrella today?",
        answer_pattern=r"umbrella",  # will fail: the polite apology has no umbrella advice
        expected_path=["model", "tools", "model"],
    ),
]


def scripts_for(case_id: str) -> dict[str, list]:
    if case_id == "api-down":
        return {"get_weather": []}  # transport has nothing: every attempt errors
    return {"get_weather": [{"temperature": 18.0, "rain_probability": 80}]}


def test_suite_runs_all_cases_and_judges_each_from_its_trace(tmp_path: Path) -> None:
    suite = run_suite(
        CASES,
        agent_factory=lambda case: weather_agent_factory(scripts_for(case.id)),
        evaluators=EVALUATORS,
        trace_dir=tmp_path,
    )

    rainy, down = suite.results

    assert rainy.passed
    assert rainy.path == ["model", "tools", "model"]

    # The api-down case DEGRADED politely: the run finished (structured error
    # observation), the trajectory was correct, but the ANSWER check caught it.
    assert not down.passed
    assert down.outcome == "finished"
    failed_checks = [check.check for check in down.checks if not check.passed]
    assert failed_checks == ["answer_matches_reference"]

    assert suite.pass_rate == 0.5
    assert not suite.all_passed


def test_every_verdict_leaves_reverifiable_evidence(tmp_path: Path) -> None:
    suite = run_suite(
        CASES[:1],
        agent_factory=lambda case: weather_agent_factory(scripts_for(case.id)),
        evaluators=EVALUATORS,
        trace_dir=tmp_path,
    )

    trace_path = Path(suite.results[0].trace_path)
    assert trace_path.exists()  # the evidence outlives the suite run


def test_a_crashing_case_is_scored_not_fatal(tmp_path: Path) -> None:
    def looping_factory(case: EvalCase) -> StateMachine:
        return StateMachine([ScriptedModelStep()], router=lambda s, c: "model", max_steps=3)

    suite = run_suite(
        [EvalCase(id="crash", query="q"), CASES[0]],
        agent_factory=lambda case: (
            looping_factory(case)
            if case.id == "crash"
            else weather_agent_factory(scripts_for(case.id))
        ),
        evaluators=EVALUATORS,
        trace_dir=tmp_path,
    )

    crash, rainy = suite.results
    assert not crash.passed
    assert crash.outcome == "failed"
    assert rainy.passed  # the suite survived the crash and kept measuring


def test_by_dimension_counts_only_applicable_checks(tmp_path: Path) -> None:
    suite = run_suite(
        CASES,
        agent_factory=lambda case: weather_agent_factory(scripts_for(case.id)),
        evaluators=EVALUATORS,
        trace_dir=tmp_path,
    )

    dims = suite.by_dimension()
    assert dims["task_completion"] == (2, 2)  # both runs finished
    assert dims["quality"] == (1, 2)  # api-down failed the answer check
    assert dims["tool_interaction"] == (2, 2)  # both took the right path


def test_scorecard_shows_verdicts_reasons_and_totals(tmp_path: Path) -> None:
    suite = run_suite(
        CASES,
        agent_factory=lambda case: weather_agent_factory(scripts_for(case.id)),
        evaluators=EVALUATORS,
        trace_dir=tmp_path,
    )

    card = render_scorecard(suite)

    assert "[PASS] rainy-day" in card
    assert "[FAIL] api-down" in card
    assert "does not match pattern" in card  # the reason, not just the verdict
    assert "model -> tools -> model" in card
    assert "overall: 1/2 cases passed (50%)" in card
