"""Tool transports: the injectable execution side of a tool call.

The registry validates INTENT; a transport performs the ACTION. Keeping the
transport behind a Protocol means the runtime never knows whether a call hit
a live API or a scripted mock -- which is exactly what lets the entire test
suite run with no network and no keys. Live transports (web search, HTTP)
arrive with the demo agent; the mock ships first because it is a first-class
citizen here, not a test hack.
"""

from typing import Any, Protocol

from pydantic import BaseModel

from agentproof.errors import TransportError
from agentproof.state import ToolCall


class ToolTransport(Protocol):
    """Executes a validated tool call and returns the raw, untrusted output.

    Raise TransportError for transient failures (timeout, rate limit) --
    the executor treats those as retryable.
    """

    def execute(self, call: ToolCall, args: BaseModel) -> Any: ...


class MockTransport:
    """Scripted transport: each tool has a queue of canned outcomes.

    An outcome may be a value (returned as the raw output) or an exception
    instance (raised) -- so tests can script "rate-limited once, then fine"
    in one line. Records every call it receives for assertions.
    """

    def __init__(self, script: dict[str, list[Any]]) -> None:
        self._script = {name: list(queue) for name, queue in script.items()}
        self.calls: list[ToolCall] = []

    def execute(self, call: ToolCall, args: BaseModel) -> Any:
        self.calls.append(call)
        queue = self._script.get(call.name)
        if not queue:
            raise TransportError(f"mock has no scripted response left for {call.name!r}")
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
