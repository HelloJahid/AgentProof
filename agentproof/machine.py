"""The execution loop: a state machine that moves AgentState through steps.

The pattern from classical computing, applied to agents: the run is a loop of
"receive state -> take one step -> produce new state". Each step does exactly
one job; the machine is the only component that knows the order, does the
bookkeeping, and enforces the step budget. That separation is what makes the
workflow predictable, testable, and modular.
"""

from collections.abc import Sequence
from typing import Protocol

from agentproof.errors import MaxStepsExceeded
from agentproof.state import AgentState


class Step(Protocol):
    """Anything with a name that transforms an AgentState.

    Steps never call each other and never decide what runs next -- they read
    the state, do their one job, and write the result back. Ordering is the
    machine's concern.
    """

    name: str

    def run(self, state: AgentState) -> AgentState: ...


class StateMachine:
    """Runs steps in sequence until the state is done or the chain ends.

    This is the linear form of the execution loop. Conditional transitions
    (choosing the next step based on what the state holds) arrive next -- the
    machine's shape is built to receive them.
    """

    def __init__(self, steps: Sequence[Step], max_steps: int = 20) -> None:
        if not steps:
            raise ValueError("StateMachine needs at least one step")
        self._steps = list(steps)
        self._max_steps = max_steps

    def run(self, state: AgentState) -> AgentState:
        executed = 0
        for step in self._steps:
            if state.is_done:
                break
            if executed >= self._max_steps:
                raise MaxStepsExceeded(self._max_steps)
            state = step.run(state)
            state.record_step(step.name)
            executed += 1
        return state
