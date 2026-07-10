"""The ModelClient port: the one door through which any LLM enters the system.

Same idea as ToolTransport, applied to the model itself: the runtime speaks
one small interface, and whether the words come from the Anthropic API or a
scripted mock is dependency injection, not a code path. The mock ships in the
package because the fully-mocked test suite is a design requirement, not a
testing shortcut.

Every completion comes back as a typed ModelResponse: the text (if any), the
tool calls (if any), and the token usage -- which the trace records and the
system-metrics eval dimension will read.
"""

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agentproof.errors import TransportError
from agentproof.state import Message, ToolCall


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ModelClient(Protocol):
    """Anything that can turn a conversation into a ModelResponse."""

    def complete(
        self,
        instructions: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse: ...


class MockModelClient:
    """Scripted model: returns queued responses in order, records every call.

    Deterministic by design -- an eval or test that uses it will produce the
    same trajectory every time, which is what makes trajectory assertions
    possible at all.
    """

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        instructions: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self.calls.append(
            {"instructions": instructions, "messages": list(messages), "tools": tools or []}
        )
        if not self._responses:
            raise TransportError("mock model has no scripted response left")
        return self._responses.pop(0)


# --- Anthropic ----------------------------------------------------------------


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Our provider-neutral conversation, in the shape the Anthropic API wants.

    - assistant turns that requested tools become tool_use content blocks;
    - tool turns become tool_result blocks inside a user turn, and consecutive
      tool turns merge into ONE user turn (the API requires alternating roles);
    - the system prompt is NOT in this list -- Anthropic takes it separately.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue  # carried via the API's system parameter
        if message.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue
        if message.role == "assistant" and message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in message.tool_calls
            )
            out.append({"role": "assistant", "content": blocks})
            continue
        out.append({"role": message.role, "content": message.content})
    return out


def parse_anthropic_response(response: Any) -> ModelResponse:
    """The API's content blocks, back into our typed ModelResponse."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
    return ModelResponse(
        text="\n".join(text_parts) or None,
        tool_calls=tool_calls,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        stop_reason=response.stop_reason,
    )


class AnthropicClient:
    """The live door: thin wrapper over the Anthropic SDK.

    Deliberately does nothing clever -- translation in, translation out. All
    behavior (retries, gates, routing) lives in the runtime where it is
    testable without a network.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_tokens: int = 1024,
        client: Any | None = None,
    ) -> None:
        if client is None:
            import anthropic  # lazy: only the live path needs the SDK ready

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def complete(
        self,
        instructions: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": to_anthropic_messages(messages),
        }
        if instructions:
            kwargs["system"] = instructions
        if tools:
            kwargs["tools"] = tools
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:  # SDK errors are transient by assumption
            raise TransportError(f"model call failed: {exc}") from exc
        return parse_anthropic_response(response)
