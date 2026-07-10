"""Phase 5: a REAL agent, end to end -- prepare -> model -> tools -> model.

The model is a MockModelClient (injection, not patching); everything else is
the production wiring. This is the first test where no step is scripted by
the test itself: the agent below is exactly what the live CLI will run, with
one constructor argument swapped.
"""

from pydantic import BaseModel, Field

from agentproof.llm import MockModelClient, ModelResponse
from agentproof.machine import StateMachine
from agentproof.state import AgentState, ToolCall
from agentproof.steps import ModelCallStep, PrepareStep, ToolExecStep, react_router
from agentproof.tools import MockTransport, Tool, ToolExecutor, ToolRegistry


class WeatherArgs(BaseModel):
    location: str = Field(min_length=2)


class WeatherReport(BaseModel):
    temperature: float
    rain_probability: int = Field(ge=0, le=100)


def build_agent(model_client: MockModelClient, transport: MockTransport) -> StateMachine:
    registry = ToolRegistry(
        [
            Tool(
                name="get_weather",
                description="Current weather for a location.",
                input_model=WeatherArgs,
                output_model=WeatherReport,
            )
        ]
    )
    return StateMachine(
        [
            PrepareStep(),
            ModelCallStep(model_client, tools=registry.specs()),
            ToolExecStep(ToolExecutor(registry, transport)),
        ],
        router=react_router,
        start="prepare",
    )


def test_real_react_agent_end_to_end_with_injected_mock_model() -> None:
    model = MockModelClient(
        [
            ModelResponse(
                text="I need the forecast.",
                tool_calls=[
                    ToolCall(id="c1", name="get_weather", arguments={"location": "Melbourne"})
                ],
                input_tokens=100,
                output_tokens=20,
            ),
            ModelResponse(
                text="18.0 degrees with an 80% chance of rain - bring an umbrella.",
                input_tokens=150,
                output_tokens=30,
            ),
        ]
    )
    transport = MockTransport({"get_weather": [{"temperature": 18.0, "rain_probability": 80}]})

    state = build_agent(model, transport).run(
        AgentState(query="Umbrella today in Melbourne?", instructions="You are a weather agent.")
    )

    assert state.final_answer == "18.0 degrees with an 80% chance of rain - bring an umbrella."
    assert [record.step for record in state.history] == ["prepare", "model", "tools", "model"]

    # The conversation tells the whole story, in order:
    roles = [message.role for message in state.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert state.messages[2].tool_calls[0].name == "get_weather"  # intent preserved
    assert state.messages[3].tool_call_id == "c1"  # observation linked to intent

    # The token bill accumulated across BOTH model calls:
    assert state.input_tokens == 250
    assert state.output_tokens == 50

    # The model was shown the tool specs on every call:
    assert model.calls[0]["tools"][0]["name"] == "get_weather"


def test_agent_answers_directly_when_the_model_wants_no_tools() -> None:
    model = MockModelClient([ModelResponse(text="Hello! How can I help?")])

    state = build_agent(model, MockTransport({})).run(AgentState(query="hi"))

    assert state.final_answer == "Hello! How can I help?"
    assert [record.step for record in state.history] == ["prepare", "model"]


def test_prepare_is_idempotent_for_resumed_conversations() -> None:
    model = MockModelClient([ModelResponse(text="answer")])
    state = AgentState(query="second question", instructions="be brief")
    state.add_message("user", "already mid-conversation")

    result = build_agent(model, MockTransport({})).run(state)

    # Prepare added nothing: the existing conversation was respected.
    assert result.messages[0].content == "already mid-conversation"
    assert all(message.role != "system" for message in result.messages[:1])
