"""Trace record schemas: what a run writes down as it happens.

Design rule: if it isn't in the trace, it didn't happen. The trace is the
single source of truth that the replay loader, the viewer, and every eval
lens consume -- none of them ever look at the live process. Each record is a
typed, versioned event; a trace file is one JSON line per event, in order.

StepCompleted carries a full snapshot of the state after the step ran. That
is deliberately redundant: it costs bytes, and buys the ability to inspect
the agent's working memory at ANY point of the run without re-executing
anything -- the property evaluation and debugging both depend on.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    seq: int  # position in the trace; strictly increasing from 0
    ts: float  # unix timestamp, seconds


class RunStarted(_Event):
    kind: Literal["run_started"] = "run_started"
    query: str
    instructions: str = ""


class StepCompleted(_Event):
    kind: Literal["step_completed"] = "step_completed"
    step: str
    duration_ms: float
    state: dict[str, Any]  # full AgentState snapshot AFTER the step ran


class RunFinished(_Event):
    kind: Literal["run_finished"] = "run_finished"
    final_answer: str | None
    steps_executed: int
    duration_ms: float


class RunFailed(_Event):
    kind: Literal["run_failed"] = "run_failed"
    error_type: str
    error_message: str
    steps_executed: int


TraceEvent = Annotated[
    RunStarted | StepCompleted | RunFinished | RunFailed,
    Field(discriminator="kind"),
]

_adapter: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)


def parse_event(line: str) -> TraceEvent:
    """One JSONL line back into its typed event -- the replay primitive."""
    return _adapter.validate_json(line)
