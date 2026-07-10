"""The execution loop: a state machine that moves AgentState through steps.

The pattern from classical computing, applied to agents: the run is a loop of
"receive state -> take one step -> produce new state". Each step does exactly
one job; the machine is the only component that knows the wiring, does the
bookkeeping, and enforces the step budget. That separation is what makes the
workflow predictable, testable, and modular.

Transitions between steps need not be fixed. A router function reads the
state after each step and chooses what runs next -- so the path taken depends
on what the state currently holds (tools requested? go execute them; answer
ready? stop). This conditional routing is what lets an agent shape its own
path through the workflow.
"""

import time
from collections.abc import Callable, Sequence
from typing import Protocol

from agentproof.errors import MaxStepsExceeded, TransitionError
from agentproof.state import AgentState
from agentproof.trace.recorder import TraceRecorder

# Given the updated state and the step that just ran, return the name of the
# next step -- or None to stop the machine.
Router = Callable[[AgentState, str], str | None]


class Step(Protocol):
    """Anything with a name that transforms an AgentState.

    Steps never call each other and never decide what runs next -- they read
    the state, do their one job, and write the result back. Ordering is the
    router's concern; bookkeeping is the machine's.
    """

    name: str

    def run(self, state: AgentState) -> AgentState: ...


def linear_router(steps: Sequence[Step]) -> Router:
    """Default routing: each step hands off to the next in the list, then stop."""
    order = [step.name for step in steps]
    successor = dict(zip(order, order[1:] + [None], strict=True))

    def route(state: AgentState, current: str) -> str | None:
        return successor[current]

    return route


class StateMachine:
    """Runs steps from a start point, following the router, until done.

    Stops when: the state carries a final answer, or the router returns None.
    Halts loudly when: the router names an unknown step (TransitionError), or
    the step budget is exhausted (MaxStepsExceeded) -- the backstop that makes
    cyclic routes (model -> tools -> model ...) safe to allow.
    """

    def __init__(
        self,
        steps: Sequence[Step],
        router: Router | None = None,
        start: str | None = None,
        max_steps: int = 20,
    ) -> None:
        if not steps:
            raise ValueError("StateMachine needs at least one step")
        names = [step.name for step in steps]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate step names: {names}")
        self._steps_by_name = {step.name: step for step in steps}
        self._router = router if router is not None else linear_router(steps)
        self._start = start if start is not None else names[0]
        self._max_steps = max_steps

    def run(self, state: AgentState, recorder: TraceRecorder | None = None) -> AgentState:
        if recorder:
            recorder.run_started(state)
        try:
            state = self._run_loop(state, recorder)
        except Exception as exc:
            if recorder:
                recorder.run_failed(exc, state)
            raise
        if recorder:
            recorder.run_finished(state)
        return state

    def _run_loop(self, state: AgentState, recorder: TraceRecorder | None) -> AgentState:
        current: str | None = self._start
        executed = 0
        while current is not None and not state.is_done:
            if executed >= self._max_steps:
                raise MaxStepsExceeded(self._max_steps)
            step = self._steps_by_name.get(current)
            if step is None:
                raise TransitionError(current)
            started = time.perf_counter()
            state = step.run(state)
            state.record_step(step.name)
            executed += 1
            if recorder:
                duration_ms = (time.perf_counter() - started) * 1000
                recorder.step_completed(step.name, state, duration_ms)
            current = self._router(state, step.name)
        return state
