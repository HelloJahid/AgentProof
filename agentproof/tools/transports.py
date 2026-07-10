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


class TavilySearchTransport:
    """Live web search via the Tavily API.

    Expects args with a `query` attribute (and optionally `max_results`).
    Returns the raw payload shaped for the tool's output gate to validate --
    the transport itself does no validation, exactly like every transport.
    The HTTP function is injectable so the request shaping is testable
    without a network.
    """

    ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key: str, post: Any | None = None, timeout: float = 15.0) -> None:
        if not api_key:
            raise ValueError("TavilySearchTransport needs an api key")
        self._api_key = api_key
        self._timeout = timeout
        if post is None:
            import httpx  # lazy: only the live path needs it

            post = httpx.post
        self._post = post

    def execute(self, call: ToolCall, args: BaseModel) -> Any:
        payload = {
            "api_key": self._api_key,
            "query": args.query,  # type: ignore[attr-defined]
            "max_results": getattr(args, "max_results", 5),
        }
        try:
            response = self._post(self.ENDPOINT, json=payload, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise TransportError(f"tavily search failed: {exc}") from exc
        return {
            "results": [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                }
                for item in data.get("results", [])
            ]
        }


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
