"""The TraceRecorder: writes the flight recorder file, one event per line.

Crash-safe by construction: every event is written and flushed the moment it
happens, so a run that dies mid-flight still leaves a readable trace up to
its last completed step. (A black box that only saves on landing would be
useless at a crash site.)
"""

import time
import uuid
from pathlib import Path
from typing import TextIO

from agentproof.state import AgentState
from agentproof.trace.records import RunFailed, RunFinished, RunStarted, StepCompleted


class TraceRecorder:
    """Records one run. Use as a context manager, or call close() yourself."""

    def __init__(self, path: Path | str, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._seq = 0
        self._file: TextIO | None = None
        self._started = time.monotonic()

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    # -- events --------------------------------------------------------------

    def run_started(self, state: AgentState) -> None:
        self._write(RunStarted(**self._head(), query=state.query, instructions=state.instructions))
        self._started = time.monotonic()

    def step_completed(self, step: str, state: AgentState, duration_ms: float) -> None:
        self._write(
            StepCompleted(
                **self._head(),
                step=step,
                duration_ms=duration_ms,
                state=state.model_dump(),
            )
        )

    def run_finished(self, state: AgentState) -> None:
        self._write(
            RunFinished(
                **self._head(),
                final_answer=state.final_answer,
                steps_executed=state.step_count,
                duration_ms=(time.monotonic() - self._started) * 1000,
            )
        )

    def run_failed(self, error: BaseException, state: AgentState) -> None:
        self._write(
            RunFailed(
                **self._head(),
                error_type=type(error).__name__,
                error_message=str(error),
                steps_executed=state.step_count,
            )
        )

    # -- internals -----------------------------------------------------------

    def _head(self) -> dict[str, object]:
        head = {"run_id": self.run_id, "seq": self._seq, "ts": time.time()}
        self._seq += 1
        return head

    def _write(self, event: RunStarted | StepCompleted | RunFinished | RunFailed) -> None:
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # "w", not "a": one file IS one run. Appending would splice two
            # runs into one file -- which load_trace rightly rejects.
            self._file = self.path.open("w", encoding="utf-8")
        self._file.write(event.model_dump_json() + "\n")
        self._file.flush()  # crash-safe: the line is on disk NOW
