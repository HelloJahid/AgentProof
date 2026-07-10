"""Phase 3: every run leaves a complete, ordered, typed trace on disk."""

from pathlib import Path

import pytest

from agentproof.errors import MaxStepsExceeded
from agentproof.machine import StateMachine
from agentproof.state import AgentState
from agentproof.trace import (
    RunFailed,
    RunFinished,
    RunStarted,
    StepCompleted,
    TraceRecorder,
    parse_event,
)


class GreetStep:
    name = "greet"

    def run(self, state: AgentState) -> AgentState:
        state.add_message("assistant", f"Hello, you asked: {state.query}")
        return state


class AnswerStep:
    name = "answer"

    def run(self, state: AgentState) -> AgentState:
        state.final_answer = state.messages[-1].content.upper()
        return state


def read_events(path: Path) -> list:
    return [parse_event(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_run_writes_start_steps_and_finish_in_order(tmp_path: Path) -> None:
    trace_path = tmp_path / "run.trace.jsonl"
    machine = StateMachine([GreetStep(), AnswerStep()])

    with TraceRecorder(trace_path) as recorder:
        machine.run(AgentState(query="what is a trace?"), recorder=recorder)

    events = read_events(trace_path)

    assert isinstance(events[0], RunStarted)
    assert events[0].query == "what is a trace?"
    assert [e.step for e in events if isinstance(e, StepCompleted)] == ["greet", "answer"]
    assert isinstance(events[-1], RunFinished)
    assert events[-1].final_answer == "HELLO, YOU ASKED: WHAT IS A TRACE?"
    assert events[-1].steps_executed == 2
    # One run id, strictly increasing sequence: the file alone tells the story.
    assert len({e.run_id for e in events}) == 1
    assert [e.seq for e in events] == list(range(len(events)))


def test_step_snapshots_show_the_state_growing(tmp_path: Path) -> None:
    trace_path = tmp_path / "run.trace.jsonl"
    machine = StateMachine([GreetStep(), AnswerStep()])

    with TraceRecorder(trace_path) as recorder:
        machine.run(AgentState(query="grow"), recorder=recorder)

    snapshots = [e for e in read_events(trace_path) if isinstance(e, StepCompleted)]

    after_greet, after_answer = snapshots
    assert after_greet.state["final_answer"] is None  # not answered yet
    assert len(after_greet.state["messages"]) == 1
    assert after_answer.state["final_answer"] is not None  # answered now
    assert after_greet.duration_ms >= 0


def test_a_crashing_run_still_leaves_its_trace(tmp_path: Path) -> None:
    trace_path = tmp_path / "crash.trace.jsonl"

    def forever(state: AgentState, current: str) -> str:
        return "greet"

    machine = StateMachine([GreetStep()], router=forever, max_steps=2)

    with TraceRecorder(trace_path) as recorder:
        with pytest.raises(MaxStepsExceeded):
            machine.run(AgentState(query="loop"), recorder=recorder)

    events = read_events(trace_path)

    # The black box survived the crash: steps up to the failure, then the failure.
    assert [e.step for e in events if isinstance(e, StepCompleted)] == ["greet", "greet"]
    assert isinstance(events[-1], RunFailed)
    assert events[-1].error_type == "MaxStepsExceeded"
    assert events[-1].steps_executed == 2


def test_runs_without_a_recorder_still_work(tmp_path: Path) -> None:
    machine = StateMachine([GreetStep(), AnswerStep()])
    state = machine.run(AgentState(query="no recorder"))
    assert state.is_done


def test_reusing_a_trace_path_overwrites_the_previous_run(tmp_path: Path) -> None:
    """One file IS one run: re-running an eval suite must not splice two
    runs into the same trace file (found live: ReplayError 'mixes events')."""
    trace_path = tmp_path / "run.trace.jsonl"
    machine = StateMachine([GreetStep(), AnswerStep()])

    with TraceRecorder(trace_path) as first:
        machine.run(AgentState(query="first run"), recorder=first)
    with TraceRecorder(trace_path) as second:
        machine.run(AgentState(query="second run"), recorder=second)

    events = read_events(trace_path)

    assert {event.run_id for event in events} == {second.run_id}  # only run two
    assert isinstance(events[0], RunStarted)
    assert events[0].query == "second run"
