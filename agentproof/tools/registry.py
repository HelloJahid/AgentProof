"""Tool declarations and the registry that validates the model's tool calls.

A Tool here is a DECLARATION, not an implementation: a name, a description
the model reads when deciding what to call, and a Pydantic input model that
defines the arguments. Execution lives elsewhere (transports) -- the same
reasoning-vs-execution separation MCP standardises: the model states intent,
the system validates it, and only then does anything actually run.

The registry is the gate on the Action side of ReAct: a ToolCall coming out
of the model is untrusted input, checked against the declared schema before
it is allowed anywhere near execution.
"""

from typing import Any

from pydantic import BaseModel, ValidationError

from agentproof.errors import ToolFailure
from agentproof.state import ToolCall


class Tool:
    """A capability the agent can advertise to the model.

    `input_model` is the contract: the model's arguments must validate
    against it, and downstream code receives a typed object, never a raw dict.
    """

    def __init__(self, name: str, description: str, input_model: type[BaseModel]) -> None:
        self.name = name
        self.description = description
        self.input_model = input_model

    def spec(self) -> dict[str, Any]:
        """The provider-agnostic description sent to the model."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }


class ToolRegistry:
    """Every tool the agent may use, and the validation gate in front of them."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolFailure(name, f"unknown tool; available: {sorted(self._tools)}")
        return tool

    def specs(self) -> list[dict[str, Any]]:
        """What the model is shown: every registered tool's declaration."""
        return [tool.spec() for tool in self._tools.values()]

    def validate_call(self, call: ToolCall) -> BaseModel:
        """Gate check a ToolCall: unknown tools and bad arguments never pass.

        On failure, the ToolFailure carries a reason written for the model,
        so a retry prompt can explain exactly what to fix.
        """
        tool = self.get(call.name)
        try:
            return tool.input_model.model_validate(call.arguments)
        except ValidationError as exc:
            raise ToolFailure(call.name, f"invalid arguments: {exc}") from exc
