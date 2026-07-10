"""Phase 2: tool declarations and the validation gate in front of execution."""

import pytest
from pydantic import BaseModel, Field

from agentproof.errors import ToolFailure
from agentproof.state import ToolCall
from agentproof.tools import Tool, ToolRegistry


class WeatherArgs(BaseModel):
    location: str = Field(min_length=2)
    units: str = "celsius"


weather_tool = Tool(
    name="get_weather",
    description="Current weather for a location.",
    input_model=WeatherArgs,
)


def test_spec_advertises_name_description_and_json_schema() -> None:
    spec = weather_tool.spec()

    assert spec["name"] == "get_weather"
    assert spec["description"] == "Current weather for a location."
    assert "location" in spec["input_schema"]["properties"]
    assert spec["input_schema"]["required"] == ["location"]


def test_registry_lists_every_tool_for_the_model() -> None:
    registry = ToolRegistry([weather_tool])
    assert [spec["name"] for spec in registry.specs()] == ["get_weather"]


def test_valid_call_returns_typed_arguments_not_a_raw_dict() -> None:
    registry = ToolRegistry([weather_tool])
    call = ToolCall(id="c1", name="get_weather", arguments={"location": "Melbourne, AU"})

    args = registry.validate_call(call)

    assert isinstance(args, WeatherArgs)
    assert args.location == "Melbourne, AU"
    assert args.units == "celsius"  # default applied


def test_unknown_tool_fails_with_the_available_tools_in_the_reason() -> None:
    registry = ToolRegistry([weather_tool])
    call = ToolCall(id="c1", name="get_wether", arguments={})  # model typo

    with pytest.raises(ToolFailure) as excinfo:
        registry.validate_call(call)

    assert excinfo.value.tool == "get_wether"
    assert "get_weather" in excinfo.value.reason  # feedback names what IS available


def test_invalid_arguments_fail_with_a_model_readable_reason() -> None:
    registry = ToolRegistry([weather_tool])
    call = ToolCall(id="c1", name="get_weather", arguments={"units": "kelvin"})

    with pytest.raises(ToolFailure) as excinfo:
        registry.validate_call(call)

    assert "location" in excinfo.value.reason  # tells the model which field to fix


def test_duplicate_tool_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        ToolRegistry([weather_tool, weather_tool])
