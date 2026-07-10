"""Phase 5 closer: the demo researcher, fully mocked -- same wiring as live."""

from types import SimpleNamespace

import pytest

from agentproof.errors import TransportError
from agentproof.llm import MockModelClient, ModelResponse
from agentproof.state import ToolCall
from agentproof.tools import MockTransport, TavilySearchTransport
from demo.researcher import INSTRUCTIONS, build_researcher, initial_state

SEARCH_PAYLOAD = {
    "results": [
        {
            "title": "NVIDIA GTC 2025 recap",
            "url": "https://techcrunch.com/gtc-2025",
            "content": "NVIDIA unveiled Blackwell Ultra at GTC 2025.",
        }
    ]
}


def test_researcher_grounds_its_answer_in_search_results() -> None:
    model = MockModelClient(
        [
            ModelResponse(
                text="Searching for the announcement.",
                tool_calls=[
                    ToolCall(
                        id="s1",
                        name="web_search",
                        arguments={"query": "NVIDIA GTC 2025 announcement"},
                    )
                ],
                input_tokens=200,
                output_tokens=25,
            ),
            ModelResponse(
                text=(
                    "NVIDIA unveiled Blackwell Ultra at GTC 2025 "
                    "[https://techcrunch.com/gtc-2025].\n\nSources:\n"
                    "- https://techcrunch.com/gtc-2025"
                ),
                input_tokens=400,
                output_tokens=60,
            ),
        ]
    )
    transport = MockTransport({"web_search": [SEARCH_PAYLOAD]})

    state = build_researcher(model, transport).run(
        initial_state("What did NVIDIA announce at GTC 2025?")
    )

    assert state.final_answer is not None
    assert "https://techcrunch.com/gtc-2025" in state.final_answer  # the citation
    assert [record.step for record in state.history] == ["prepare", "model", "tools", "model"]
    # The system turn carries the grounding rules the agent lives by:
    assert state.messages[0].role == "system"
    assert "cite its source URL" in state.messages[0].content
    assert INSTRUCTIONS in state.messages[0].content


def test_search_output_must_pass_the_observation_gate() -> None:
    model = MockModelClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="s1", name="web_search", arguments={"query": "anything at all"})
                ]
            ),
            ModelResponse(text="I could not find reliable information."),
        ]
    )
    # Malformed payload every attempt: the gate rejects it, retries exhaust.
    transport = MockTransport({"web_search": [{"nonsense": True}] * 3})

    state = build_researcher(model, transport).run(initial_state("anything?"))

    # The agent survived and answered honestly from a structured error:
    assert state.tool_results[0].is_error
    assert state.final_answer == "I could not find reliable information."


# --- the live transport's request shaping, without a network -------------------


class FakeArgs:
    query = "test query"
    max_results = 3


def test_tavily_transport_shapes_the_request_and_the_payload() -> None:
    seen: dict = {}

    def fake_post(url: str, json: dict, timeout: float) -> SimpleNamespace:
        seen.update({"url": url, "json": json, "timeout": timeout})
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "results": [{"title": "T", "url": "https://x", "content": "C", "extra": "dropped"}]
            },
        )

    transport = TavilySearchTransport("key-123", post=fake_post)
    raw = transport.execute(ToolCall(id="c", name="web_search"), FakeArgs())  # type: ignore[arg-type]

    assert seen["url"] == TavilySearchTransport.ENDPOINT
    assert seen["json"] == {"api_key": "key-123", "query": "test query", "max_results": 3}
    assert raw == {"results": [{"title": "T", "url": "https://x", "content": "C"}]}


def test_tavily_transport_wraps_http_failures_as_transient() -> None:
    def failing_post(url: str, json: dict, timeout: float) -> SimpleNamespace:
        raise ConnectionError("boom")

    transport = TavilySearchTransport("key-123", post=failing_post)

    with pytest.raises(TransportError, match="tavily search failed"):
        transport.execute(ToolCall(id="c", name="web_search"), FakeArgs())  # type: ignore[arg-type]


def test_tavily_transport_requires_a_key() -> None:
    with pytest.raises(ValueError):
        TavilySearchTransport("")
