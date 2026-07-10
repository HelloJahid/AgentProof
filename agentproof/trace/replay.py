"""Replay: reconstruct a complete run from its trace file alone.

This is the other half of the flight-recorder promise. The recorder wrote
events as they happened; the loader turns that file back into a structured,
typed account of the run -- without re-executing anything. Every eval lens
and the viewer consume a RunReplay, never the live process.

A trace that fails integrity checks (wrong first event, mixed run ids, gaps
in the sequence) is rejected with ReplayError: evidence you cannot trust is
worse than no evidence. A trace that simply STOPS (process killed mid-run)
is still loadable -- outcome "truncated" -- because a crash is precisely
when the black box matters most.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from agentproof.errors import ReplayError
from agentproof.state import AgentState
from agentproof.trace.records import (
    RunFailed,
    RunFinished,
    RunStarted,
    StepCompleted,
    parse_event,
)


class ReplayStep(BaseModel):
    """One executed step: what ran, how long it took, what the state became."""

    model_config = ConfigDict(extra="forbid")

    name: str
    duration_ms: float
    state: AgentState  # the snapshot, back as a fully typed object


class RunReplay(BaseModel):
    """The whole run, reconstructed: the input every evaluator consumes."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    query: str
    instructions: str
    steps: list[ReplayStep]
    outcome: Literal["finished", "failed", "truncated"]
    final_answer: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    total_duration_ms: float | None = None

    @property
    def path(self) -> list[str]:
        """The trajectory: the ordered step names the run actually took."""
        return [step.name for step in self.steps]

    @property
    def final_state(self) -> AgentState | None:
        return self.steps[-1].state if self.steps else None


def load_trace(path: Path | str) -> RunReplay:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ReplayError("trace file is empty")

    try:
        events = [parse_event(line) for line in lines]
    except ValueError as exc:
        raise ReplayError(f"unparseable event: {exc}") from exc

    head = events[0]
    if not isinstance(head, RunStarted):
        raise ReplayError(f"first event must be run_started, got {head.kind!r}")
    if len({event.run_id for event in events}) != 1:
        raise ReplayError("trace mixes events from more than one run")
    if [event.seq for event in events] != list(range(len(events))):
        raise ReplayError("sequence numbers have gaps or are out of order")

    steps = [
        ReplayStep(
            name=event.step,
            duration_ms=event.duration_ms,
            state=AgentState.model_validate(event.state),
        )
        for event in events
        if isinstance(event, StepCompleted)
    ]

    replay = RunReplay(
        run_id=head.run_id,
        query=head.query,
        instructions=head.instructions,
        steps=steps,
        outcome="truncated",
    )

    tail = events[-1]
    if isinstance(tail, RunFinished):
        replay.outcome = "finished"
        replay.final_answer = tail.final_answer
        replay.total_duration_ms = tail.duration_ms
    elif isinstance(tail, RunFailed):
        replay.outcome = "failed"
        replay.error_type = tail.error_type
        replay.error_message = tail.error_message

    return replay
