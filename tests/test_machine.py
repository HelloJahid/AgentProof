"""Phase 1: the StateMachine drives state through steps, with a step budget."""

import pytest

from agentproof.errors import MaxStepsExceeded, TransitionError
from agentproof.machine import StateMachine
from agentproof.state import AgentState, ToolCall, ToolResult


class GreetStep:
    """Toy step: writes a greeting message into the state."""

    name = "greet"

    def run(self, state: AgentState) -> AgentState:
        state.add_message("assistant", f"Hello, you asked: {state.query}")
        return state


class AnswerStep:
    """Toy step: produces the final answer, ending the run."""

    name = "answer"

    def run(self, state: AgentState) -> AgentState:
        state.final_answer = state.messages[-1].content.upper()
        return state


def test_two_step_machine_runs_end_to_end() -> None:
    machine = StateMachine([GreetStep(), AnswerStep()])
    state = machine.run(AgentState(query="what is state?"))

    assert state.final_answer == "HELLO, YOU ASKED: WHAT IS STATE?"
    assert [record.step for record in state.history] == ["greet", "answer"]
    assert state.step_count == 2


def test_machine_stops_as_soon_as_state_is_done() -> None:
    class EagerAnswer:
        name = "eager"

        def run(self, state: AgentState) -> AgentState:
            state.final_answer = "done immediately"
            return state

    class NeverRuns:
        name = "never"

        def run(self, state: AgentState) -> AgentState:  # pragma: no cover
            raise AssertionError("machine should have stopped before this step")

    state = StateMachine([EagerAnswer(), NeverRuns()]).run(AgentState(query="hi"))

    assert state.final_answer == "done immediately"
    assert [record.step for record in state.history] == ["eager"]


def test_step_budget_halts_a_run_that_wont_finish() -> None:
    machine = StateMachine([GreetStep(), AnswerStep()], max_steps=1)

    with pytest.raises(MaxStepsExceeded) as excinfo:
        machine.run(AgentState(query="hi"))

    assert excinfo.value.limit == 1


def test_machine_requires_at_least_one_step() -> None:
    with pytest.raises(ValueError):
        StateMachine([])


def test_work_done_early_stays_visible_to_later_steps() -> None:
    """The whole point of shared state: step 2 reads what step 1 wrote."""
    machine = StateMachine([GreetStep(), AnswerStep()])
    state = machine.run(AgentState(query="carry me forward"))

    # AnswerStep built its answer from the message GreetStep left behind.
    assert "CARRY ME FORWARD" in state.final_answer


# --- Conditional transitions -------------------------------------------------


class ToyModelStep:
    """First pass: request a tool. Second pass (tool result present): answer."""

    name = "model"

    def run(self, state: AgentState) -> AgentState:
        if state.tool_results:
            observation = state.tool_results[-1].output
            state.final_answer = f"Based on the tool: {observation}"
        else:
            state.pending_tool_calls.append(
                ToolCall(id="call_1", name="lookup", arguments={"q": state.query})
            )
        return state


class ToyToolStep:
    """Executes the pending tool call and records the observation."""

    name = "tools"

    def run(self, state: AgentState) -> AgentState:
        call = state.pending_tool_calls.pop()
        state.tool_results.append(ToolResult(call_id=call.id, name=call.name, output="42"))
        return state


def react_router(state: AgentState, current: str) -> str | None:
    """Tools requested -> run them; after tools -> back to the model."""
    if state.pending_tool_calls:
        return "tools"
    if current == "tools":
        return "model"
    return None


def test_conditional_routing_lets_the_run_cycle_model_tools_model() -> None:
    machine = StateMachine([ToyModelStep(), ToyToolStep()], router=react_router, start="model")
    state = machine.run(AgentState(query="what is the answer?"))

    assert state.final_answer == "Based on the tool: 42"
    # The path taken depended on the state: model asked -> tools ran -> model answered.
    assert [record.step for record in state.history] == ["model", "tools", "model"]


def test_router_naming_an_unknown_step_raises_transition_error() -> None:
    def bad_router(state: AgentState, current: str) -> str | None:
        return "nonexistent_step"

    machine = StateMachine([GreetStep(), AnswerStep()], router=bad_router)

    with pytest.raises(TransitionError) as excinfo:
        machine.run(AgentState(query="hi"))

    assert excinfo.value.target == "nonexistent_step"


def test_a_cyclic_route_that_never_finishes_hits_the_step_budget() -> None:
    def forever_router(state: AgentState, current: str) -> str | None:
        return "greet"  # loop on the same step, never producing an answer

    machine = StateMachine([GreetStep()], router=forever_router, max_steps=5)

    with pytest.raises(MaxStepsExceeded):
        machine.run(AgentState(query="hi"))


def test_duplicate_step_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        StateMachine([GreetStep(), GreetStep()])
