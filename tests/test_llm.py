"""Phase 5: the ModelClient port -- mock behavior and Anthropic translation.

No test here touches the network. The live client is a thin wrapper; what we
test is everything it delegates to: the pure translation functions.
"""

from types import SimpleNamespace

import pytest

from agentproof.errors import TransportError
from agentproof.llm import (
    MockModelClient,
    ModelResponse,
    parse_anthropic_response,
    to_anthropic_messages,
)
from agentproof.state import Message, ToolCall

# --- mock client ---------------------------------------------------------------


def test_mock_client_replays_its_script_in_order_and_records_calls() -> None:
    client = MockModelClient(
        [
            ModelResponse(tool_calls=[ToolCall(id="c1", name="lookup", arguments={})]),
            ModelResponse(text="final answer"),
        ]
    )

    first = client.complete("be helpful", [Message(role="user", content="hi")])
    second = client.complete("be helpful", [Message(role="user", content="hi")])

    assert first.wants_tools and not second.wants_tools
    assert second.text == "final answer"
    assert len(client.calls) == 2
    assert client.calls[0]["instructions"] == "be helpful"


def test_mock_client_fails_loudly_when_the_script_runs_out() -> None:
    client = MockModelClient([])
    with pytest.raises(TransportError, match="no scripted response"):
        client.complete("", [])


# --- translation to Anthropic shape ---------------------------------------------


def test_plain_turns_pass_through_and_system_is_excluded() -> None:
    messages = [
        Message(role="system", content="be helpful"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi there"),
    ]

    out = to_anthropic_messages(messages)

    assert out == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_assistant_tool_requests_become_tool_use_blocks() -> None:
    messages = [
        Message(
            role="assistant",
            content="checking the weather",
            tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"location": "Mel"})],
        )
    ]

    (turn,) = to_anthropic_messages(messages)

    assert turn["role"] == "assistant"
    assert turn["content"][0] == {"type": "text", "text": "checking the weather"}
    assert turn["content"][1]["type"] == "tool_use"
    assert turn["content"][1]["id"] == "c1"
    assert turn["content"][1]["input"] == {"location": "Mel"}


def test_consecutive_tool_results_merge_into_one_user_turn() -> None:
    messages = [
        Message(role="tool", content='{"t": 18}', tool_call_id="c1"),
        Message(role="tool", content='{"h": 60}', tool_call_id="c2"),
    ]

    (turn,) = to_anthropic_messages(messages)

    assert turn["role"] == "user"
    assert [block["tool_use_id"] for block in turn["content"]] == ["c1", "c2"]
    assert all(block["type"] == "tool_result" for block in turn["content"])


# --- parsing the Anthropic response ---------------------------------------------


def fake_response(*blocks: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        usage=SimpleNamespace(input_tokens=120, output_tokens=45),
        stop_reason="tool_use",
    )


def test_response_blocks_parse_into_a_typed_model_response() -> None:
    response = fake_response(
        SimpleNamespace(type="text", text="Let me check."),
        SimpleNamespace(
            type="tool_use", id="c9", name="get_weather", input={"location": "Melbourne"}
        ),
    )

    parsed = parse_anthropic_response(response)

    assert parsed.text == "Let me check."
    assert parsed.wants_tools
    assert parsed.tool_calls[0] == ToolCall(
        id="c9", name="get_weather", arguments={"location": "Melbourne"}
    )
    assert parsed.input_tokens == 120 and parsed.output_tokens == 45
    assert parsed.stop_reason == "tool_use"
