"""Phase 2: the executor turns untrusted tool activity into trustworthy
observations -- or structured errors the agent can reason about."""

from pydantic import BaseModel, Field

from agentproof.errors import TransportError
from agentproof.state import ToolCall
from agentproof.tools import MockTransport, Tool, ToolExecutor, ToolRegistry


class WeatherArgs(BaseModel):
    location: str = Field(min_length=2)


class WeatherReport(BaseModel):
    temperature: float = Field(ge=-50, le=60)
    rain_probability: int = Field(ge=0, le=100)


def make_weather_tool() -> Tool:
    return Tool(
        name="get_weather",
        description="Current weather for a location.",
        input_model=WeatherArgs,
        output_model=WeatherReport,
    )


GOOD_PAYLOAD = {"temperature": 18.0, "rain_probability": 80}
CALL = ToolCall(id="c1", name="get_weather", arguments={"location": "Melbourne, AU"})


def test_valid_output_passes_the_observation_gate() -> None:
    transport = MockTransport({"get_weather": [GOOD_PAYLOAD]})
    executor = ToolExecutor(ToolRegistry([make_weather_tool()]), transport)

    result = executor.execute_call(CALL)

    assert not result.is_error
    assert '"temperature":18.0' in result.output
    assert result.call_id == "c1"


def test_transient_failures_are_absorbed_by_retry() -> None:
    transport = MockTransport(
        {
            "get_weather": [
                TransportError("rate_limit_exceeded"),  # attempt 1: API says no
                {"error": "oops"},  # attempt 2: malformed payload
                GOOD_PAYLOAD,  # attempt 3: fine
            ]
        }
    )
    executor = ToolExecutor(ToolRegistry([make_weather_tool()]), transport, max_attempts=3)

    result = executor.execute_call(CALL)

    assert not result.is_error
    assert len(transport.calls) == 3  # the agent never saw attempts 1 and 2


def test_exhausted_retries_return_a_structured_error_not_an_exception() -> None:
    transport = MockTransport(
        {"get_weather": [TransportError("timeout")] * 3},
    )
    executor = ToolExecutor(ToolRegistry([make_weather_tool()]), transport, max_attempts=3)

    result = executor.execute_call(CALL)

    assert result.is_error
    assert "after 3 attempts" in result.output
    assert "timeout" in result.output  # the agent can reason about why


def test_bad_arguments_never_reach_the_transport() -> None:
    transport = MockTransport({"get_weather": [GOOD_PAYLOAD]})
    executor = ToolExecutor(ToolRegistry([make_weather_tool()]), transport)
    bad_call = ToolCall(id="c2", name="get_weather", arguments={})  # missing location

    result = executor.execute_call(bad_call)

    assert result.is_error
    assert "location" in result.output  # feedback names the field to fix
    assert transport.calls == []  # the world was never touched


def test_json_string_output_is_parsed_then_gated() -> None:
    transport = MockTransport({"get_weather": ['{"temperature": 18.0, "rain_probability": 80}']})
    executor = ToolExecutor(ToolRegistry([make_weather_tool()]), transport)

    result = executor.execute_call(CALL)

    assert not result.is_error


def test_tools_without_an_output_model_pass_raw_output_through() -> None:
    notes_tool = Tool(name="read_note", description="Read a note.", input_model=WeatherArgs)
    transport = MockTransport({"read_note": ["just some text"]})
    executor = ToolExecutor(ToolRegistry([notes_tool]), transport)

    result = executor.execute_call(
        ToolCall(id="c3", name="read_note", arguments={"location": "anywhere"})
    )

    assert not result.is_error
    assert result.output == "just some text"
