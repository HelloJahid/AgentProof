"""Phase 5: memory policies shape what the model sees, never what the run knows."""

import pytest

from agentproof.llm import MockModelClient, ModelResponse
from agentproof.memory import FullHistory, SlidingWindow, SummarizingMemory
from agentproof.state import AgentState, Message
from agentproof.steps import ModelCallStep


def conversation() -> list[Message]:
    turns: list[Message] = [Message(role="system", content="be helpful")]
    for i in range(1, 7):  # 12 non-system turns: u1,a1,u2,a2,...
        turns.append(Message(role="user", content=f"question {i}"))
        turns.append(Message(role="assistant", content=f"answer {i}"))
    return turns


def test_full_history_sends_everything() -> None:
    messages = conversation()
    assert FullHistory().view(messages) == messages


def test_sliding_window_keeps_system_plus_the_recent_tail() -> None:
    view = SlidingWindow(max_turns=4).view(conversation())

    assert view[0].role == "system"  # identity survives any window
    assert [message.content for message in view[1:]] == [
        "question 5",
        "answer 5",
        "question 6",
        "answer 6",
    ]


def test_sliding_window_never_starts_with_an_orphan_observation() -> None:
    messages = [
        Message(role="user", content="q"),
        Message(role="assistant", content="calling tool"),
        Message(role="tool", content="result", tool_call_id="c1"),
        Message(role="assistant", content="done"),
    ]

    # A window of 2 would start at the tool result -- an observation whose
    # intent was cut off. The policy drops it rather than confuse the model.
    view = SlidingWindow(max_turns=2).view(messages)

    assert [message.role for message in view] == ["assistant"]
    assert view[0].content == "done"


def test_summarizing_memory_compresses_the_old_and_keeps_the_recent() -> None:
    summarizer = MockModelClient([ModelResponse(text="Earlier: questions 1-4 were answered.")])
    view = SummarizingMemory(summarizer, keep_recent=4).view(conversation())

    assert view[0].role == "system"
    assert "[Summary of the earlier conversation]" in view[1].content
    assert "questions 1-4" in view[1].content
    assert [message.content for message in view[2:]] == [
        "question 5",
        "answer 5",
        "question 6",
        "answer 6",
    ]
    # The summarizer was shown the OLD turns as a transcript:
    transcript = summarizer.calls[0]["messages"][0].content
    assert "question 1" in transcript and "question 4" in transcript
    assert "question 6" not in transcript


def test_summarizing_memory_is_a_noop_below_the_threshold() -> None:
    summarizer = MockModelClient([])  # would raise if ever called
    messages = conversation()[:5]

    assert SummarizingMemory(summarizer, keep_recent=6).view(messages) == messages
    assert summarizer.calls == []


def test_model_call_applies_the_policy_but_the_state_keeps_the_truth() -> None:
    model = MockModelClient([ModelResponse(text="final")])
    step = ModelCallStep(model, memory=SlidingWindow(max_turns=2))

    state = AgentState(query="q")
    state.messages = conversation()
    before = len(state.messages)

    state = step.run(state)

    # The model saw the window (system + 2 turns)...
    assert len(model.calls[0]["messages"]) == 3
    # ...but the state's record only GREW (full truth, plus the new answer).
    assert len(state.messages) == before + 1


def test_policies_reject_nonsense_configuration() -> None:
    with pytest.raises(ValueError):
        SlidingWindow(max_turns=0)
    with pytest.raises(ValueError):
        SummarizingMemory(MockModelClient([]), keep_recent=0)
