"""Typed errors: every failure in AgentProof has a name and a reason.

Convention: never raise bare Exception. A typed error tells the caller (and
the future trace recorder) exactly which kind of failure occurred, so it can
be handled, retried, or reported deliberately.
"""


class AgentProofError(Exception):
    """Base class for all AgentProof errors."""


class TransitionError(AgentProofError):
    """The router chose a step the machine does not know.

    A transition target must always be a registered step name (or None to
    stop). Anything else is a wiring bug, and the machine halts loudly at the
    exact moment the bad decision was made rather than misbehaving later.
    """

    def __init__(self, target: str) -> None:
        self.target = target
        super().__init__(f"transition to unknown step: {target!r}")


class MaxStepsExceeded(AgentProofError):
    """The state machine hit its step budget -- a runaway-loop backstop.

    An agent loop must never mean "forever": if the machine is still running
    after `limit` steps, something is wrong (a cycle, a model that never
    finishes), and halting loudly beats burning tokens quietly.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"state machine exceeded max_steps={limit}")
