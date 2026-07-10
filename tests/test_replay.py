"""Phase 3 closer: a run is fully reconstructable from its trace file alone."""

from pathlib import Path

import pytest

from agentproof.errors import MaxStepsExceeded, ReplayError
from agentproof.machine import StateMachine
from agentproof.state import AgentState
from agentproof.trace import TraceRecorder, load_trace


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


def record_run(tmp_path: Path) -> Path:
    trace_path = tmp_path / "run.trace.jsonl"
    machine = StateMachine([GreetStep(), AnswerStep()])
    with TraceRecorder(trace_path) as recorder:
        machine.run(AgentState(query="replay me"), recorder=recorder)
    return trace_path


def test_a_finished_run_replays_completely_from_the_file(tmp_path: Path) -> None:
    replay = load_trace(record_run(tmp_path))

    assert replay.outcome == "finished"
    assert replay.query == "replay me"
    assert replay.path == ["greet", "answer"]
    assert replay.final_answer == "HELLO, YOU ASKED: REPLAY ME"
    # Snapshots come back as fully typed AgentState, not dicts:
    assert replay.steps[0].state.messages[0].role == "assistant"
    assert replay.final_state is not None and replay.final_state.is_done


def test_a_failed_run_replays_with_its_error(tmp_path: Path) -> None:
    trace_path = tmp_path / "crash.trace.jsonl"
    machine = StateMachine([GreetStep()], router=lambda s, c: "greet", max_steps=2)

    with TraceRecorder(trace_path) as recorder:
        with pytest.raises(MaxStepsExceeded):
            machine.run(AgentState(query="loop"), recorder=recorder)

    replay = load_trace(trace_path)

    assert replay.outcome == "failed"
    assert replay.error_type == "MaxStepsExceeded"
    assert replay.path == ["greet", "greet"]


def test_a_killed_process_leaves_a_truncated_but_loadable_trace(tmp_path: Path) -> None:
    trace_path = record_run(tmp_path)
    # Simulate the process dying mid-run: drop everything after the first step.
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    trace_path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")

    replay = load_trace(trace_path)

    assert replay.outcome == "truncated"
    assert replay.path == ["greet"]  # the story up to the moment of death
    assert replay.final_answer is None


def test_corrupt_traces_are_rejected_not_half_loaded(tmp_path: Path) -> None:
    trace_path = record_run(tmp_path)
    lines = trace_path.read_text(encoding="utf-8").splitlines()

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ReplayError, match="empty"):
        load_trace(empty)

    headless = tmp_path / "headless.jsonl"
    headless.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="run_started"):
        load_trace(headless)

    gappy = tmp_path / "gappy.jsonl"
    gappy.write_text("\n".join([lines[0]] + lines[2:]) + "\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="sequence"):
        load_trace(gappy)
