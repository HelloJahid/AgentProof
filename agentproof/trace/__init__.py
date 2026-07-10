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
from agentproof.trace.replay import ReplayStep, RunReplay, load_trace
from agentproof.trace.viewer import render_html, render_text, write_html

__all__ = [
    "ReplayStep",
    "RunFailed",
    "RunFinished",
    "RunReplay",
    "RunStarted",
    "StepCompleted",
    "TraceEvent",
    "TraceRecorder",
    "load_trace",
    "render_html",
    "render_text",
    "write_html",
    "parse_event",
]
