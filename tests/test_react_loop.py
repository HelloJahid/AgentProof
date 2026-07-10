"""Phase 2 closer: the full ReAct cycle through the real machine.

A scripted model step stands in for the LLM (the real model step arrives with
the ModelClient). Everything else is the real thing: StateMachine, router,
registry, gates, retrying executor, mock transport.
"""

from pydantic import BaseModel, Field

from agentproof.errors import TransportError
from agentproof.machine import StateMachine
from agentproof.state import AgentState, ToolCall
from agentproof.steps import ToolExecStep, react_router
from agentproof.tools import MockTransport, Tool, ToolExecutor, ToolRegistry


class WeatherArgs(BaseModel):
    location: str = Field(min_length=2)


class WeatherReport(BaseModel):
    temperature: float
    rain_probability: int = Field(ge=0, le=100)


class ScriptedModelStep:
    """Stands in for the LLM: first asks for the weather, then answers from
    the observation -- the Thought half of ReAct, scripted."""

    name = "model"

    def run(self, state: AgentState) -> AgentState:
        if not state.tool_results:
            state.pending_tool_calls.append(
                ToolCall(
                    id="call_1",
                    name="get_weather",
                    arguments={"location": "Melbourne, AU"},
                )
            )
            return state

        observation = WeatherReport.model_validate_json(state.tool_results[-1].output)
        advice = "bring an umbrella" if observation.rain_probability > 50 else "no umbrella needed"
        state.final_answer = f"{observation.temperature} degrees, {advice}."
        return state


def build_machine(transport: MockTransport) -> StateMachine:
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
    executor = ToolExecutor(registry, transport, max_attempts=3)
    return StateMachine(
        [ScriptedModelStep(), ToolExecStep(executor)],
        router=react_router,
        start="model",
    )


def test_full_react_cycle_model_tools_model() -> None:
    transport = MockTransport({"get_weather": [{"temperature": 18.0, "rain_probability": 80}]})

    state = build_machine(transport).run(AgentState(query="Umbrella today in Melbourne?"))

    assert state.final_answer == "18.0 degrees, bring an umbrella."
    assert [record.step for record in state.history] == ["model", "tools", "model"]
    assert state.pending_tool_calls == []  # every wish was consumed
    assert state.tool_results[0].name == "get_weather"
    assert state.messages[-1].role == "tool"  # the observation joined the conversation


def test_react_cycle_survives_a_rate_limited_first_attempt() -> None:
    transport = MockTransport(
        {
            "get_weather": [
                TransportError("rate_limit_exceeded"),
                {"temperature": 18.0, "rain_probability": 80},
            ]
        }
    )

    state = build_machine(transport).run(AgentState(query="Umbrella today?"))

    # The retry happened INSIDE the tools step: the path the agent took is
    # identical to the happy case, and the reasoning chain never saw the failure.
    assert state.final_answer == "18.0 degrees, bring an umbrella."
    assert [record.step for record in state.history] == ["model", "tools", "model"]
    assert len(transport.calls) == 2
    assert not state.tool_results[0].is_error
