"""Phase 1: the AgentState schema behaves as typed, validated working memory."""

import pytest
from pydantic import ValidationError

from agentproof.state import AgentState, Message, ToolCall, ToolResult


def test_state_starts_empty_except_for_the_query() -> None:
    state = AgentState(query="What is the weather in Melbourne?")

    assert state.query == "What is the weather in Melbourne?"
    assert state.messages == []
    assert state.pending_tool_calls == []
    assert state.tool_results == []
    assert state.history == []
    assert state.final_answer is None
    assert state.step_count == 0
    assert not state.is_done


def test_unknown_fields_are_rejected_so_mismatches_surface_early() -> None:
    with pytest.raises(ValidationError):
        AgentState(query="hi", nonexistent_field="oops")  # type: ignore[call-arg]


def test_steps_accumulate_messages_and_history() -> None:
    state = AgentState(query="hi", instructions="You are a helpful agent.")

    state.add_message("system", state.instructions)
    state.add_message("user", state.query)
    state.record_step("prepare", "built initial prompt")

    assert state.messages == [
        Message(role="system", content="You are a helpful agent."),
        Message(role="user", content="hi"),
    ]
    assert state.step_count == 1
    assert state.history[0].step == "prepare"


def test_tool_activity_is_tracked_as_intent_then_observation() -> None:
    state = AgentState(query="weather?")

    state.pending_tool_calls.append(
        ToolCall(id="call_1", name="get_weather", arguments={"location": "Melbourne, AU"})
    )
    state.tool_results.append(
        ToolResult(call_id="call_1", name="get_weather", output='{"temperature": 18}')
    )

    assert state.pending_tool_calls[0].arguments["location"] == "Melbourne, AU"
    assert state.tool_results[0].call_id == state.pending_tool_calls[0].id
    assert not state.tool_results[0].is_error


def test_a_final_answer_marks_the_run_done() -> None:
    state = AgentState(query="hi")
    state.final_answer = "Hello!"
    assert state.is_done


def test_two_states_never_share_mutable_defaults() -> None:
    a = AgentState(query="a")
    b = AgentState(query="b")
    a.add_message("user", "only in a")

    assert b.messages == []
