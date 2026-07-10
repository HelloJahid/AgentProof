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


class GateFailure(AgentProofError):
    """An output failed validation at a gate check.

    Gates sit between steps (and between Action and Observation) so that
    malformed data is caught at the boundary where it appears, never passed
    downstream to corrupt the reasoning chain.
    """

    def __init__(self, where: str, reason: str) -> None:
        self.where = where
        self.reason = reason
        super().__init__(f"gate check failed at {where}: {reason}")


class TransportError(AgentProofError):
    """A tool transport failed to produce a response (timeout, rate limit,
    network). Transient by assumption: the executor may retry it."""


class ToolFailure(AgentProofError):
    """A tool call could not be validated or executed.

    Carries a model-readable `reason` so a retry can include what went wrong
    -- the "retry with feedback" pattern: telling the model why its last
    attempt failed makes the next attempt less likely to repeat the mistake.
    """

    def __init__(self, tool: str, reason: str) -> None:
        self.tool = tool
        self.reason = reason
        super().__init__(f"tool {tool!r} failed: {reason}")


class ReplayError(AgentProofError):
    """A trace file cannot be trusted: corrupt, out of order, or not a trace.

    A replay is evidence -- evals and debugging both reason from it -- so a
    file that fails integrity checks is rejected loudly rather than loaded
    into a half-true story.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"cannot replay trace: {reason}")


class MaxStepsExceeded(AgentProofError):
    """The state machine hit its step budget -- a runaway-loop backstop.

    An agent loop must never mean "forever": if the machine is still running
    after `limit` steps, something is wrong (a cycle, a model that never
    finishes), and halting loudly beats burning tokens quietly.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"state machine exceeded max_steps={limit}")
