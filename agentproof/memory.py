"""Short-term memory: what the model SEES, not what the run KNOWS.

A memory policy is a VIEW over the conversation, applied at the moment of a
model call. The state keeps every message (the flight recorder depends on
that); the policy only decides how much of it rides along in the next
prompt. That separation is the whole design: forgetting is a presentation
choice, never a loss of evidence.

Three strategies, three trade-offs (straight from the notes):
  full history   -- maximum fidelity, growing token bill;
  sliding window -- bounded cost, may drop something important;
  summarisation  -- compact, but only as good as the summary.
"""

from typing import Protocol

from agentproof.llm import ModelClient
from agentproof.state import Message


class MemoryPolicy(Protocol):
    """Decides which view of the conversation the next model call sees."""

    def view(self, messages: list[Message]) -> list[Message]: ...


def _split_system(messages: list[Message]) -> tuple[list[Message], list[Message]]:
    """Leading system turns are identity, not history -- always kept."""
    head = 0
    while head < len(messages) and messages[head].role == "system":
        head += 1
    return list(messages[:head]), list(messages[head:])


def _drop_orphan_tools(messages: list[Message]) -> list[Message]:
    """A window must not START with tool results whose request was cut off:
    an observation without its intent is meaningless to the model (and
    invalid for providers that link results to tool_use ids)."""
    start = 0
    while start < len(messages) and messages[start].role == "tool":
        start += 1
    return messages[start:]


class FullHistory:
    """Send everything. Maximum detail, maximum tokens."""

    def view(self, messages: list[Message]) -> list[Message]:
        return list(messages)


class SlidingWindow:
    """Send the system turns plus only the last `max_turns` messages."""

    def __init__(self, max_turns: int = 10) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self._max_turns = max_turns

    def view(self, messages: list[Message]) -> list[Message]:
        system, rest = _split_system(messages)
        window = _drop_orphan_tools(rest[-self._max_turns :])
        return system + window


class SummarizingMemory:
    """Compress everything older than the recent tail into one summary turn.

    The summary is produced by a model call through the same ModelClient
    port as everything else -- so in tests the summarizer is a scripted
    mock, and the strategy stays fully deterministic.
    """

    _INSTRUCTIONS = (
        "You compress conversations. Summarise the following exchange in a "
        "short paragraph, keeping every fact, decision, and open question. "
        "Reply with the summary only."
    )

    def __init__(self, client: ModelClient, keep_recent: int = 6) -> None:
        if keep_recent < 1:
            raise ValueError("keep_recent must be at least 1")
        self._client = client
        self._keep_recent = keep_recent

    def view(self, messages: list[Message]) -> list[Message]:
        system, rest = _split_system(messages)
        if len(rest) <= self._keep_recent:
            return list(messages)

        old = rest[: -self._keep_recent]
        recent = _drop_orphan_tools(rest[-self._keep_recent :])

        transcript = "\n".join(f"{message.role}: {message.content}" for message in old)
        response = self._client.complete(
            self._INSTRUCTIONS, [Message(role="user", content=transcript)]
        )
        summary = Message(
            role="user",
            content=f"[Summary of the earlier conversation]\n{response.text or ''}",
        )
        return system + [summary] + recent
