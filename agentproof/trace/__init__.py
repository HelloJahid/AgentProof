"""The flight recorder: every run leaves a complete, replayable trace."""

from agentproof.trace.recorder import TraceRecorder
from agentproof.trace.records import (
    RunFailed,
    RunFinished,
    RunStarted,
    StepCompleted,
    TraceEvent,
    parse_event,
)

__all__ = [
    "RunFailed",
    "RunFinished",
    "RunStarted",
    "StepCompleted",
    "TraceEvent",
    "TraceRecorder",
    "parse_event",
]
