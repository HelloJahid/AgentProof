"""Phase 1: the StateMachine drives state through steps, with a step budget."""

import pytest

from agentproof.errors import MaxStepsExceeded
from agentproof.machine import StateMachine
from agentproof.state import AgentState


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
