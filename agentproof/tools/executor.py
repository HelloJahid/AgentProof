"""The gate-checked tool executor: intent in, trustworthy observation out.

External tools fail far more often than the model does: rate limits, timeouts,
malformed payloads, schema drift. The executor is the machinery that absorbs
those failures so the reasoning chain only ever receives one of two things --
a validated observation, or a structured error it can reason about. Never a
raw traceback, never a half-broken payload.

The flow per call:
  1. Action gate: arguments validated against the tool's input model.
     Failure here is the MODEL's mistake -- retrying the transport cannot fix
     it, so the feedback goes straight back as an error observation.
  2. Transport executes (transient failures raised as TransportError).
  3. Observation gate: raw output validated against the tool's output model.
     Failures at 2 or 3 are the WORLD's mistake -- transient by assumption,
     so the executor retries up to max_attempts before giving up.
"""

import json

from pydantic import ValidationError

from agentproof.errors import GateFailure, ToolFailure, TransportError
from agentproof.state import ToolCall, ToolResult
from agentproof.tools.registry import Tool, ToolRegistry
from agentproof.tools.transports import ToolTransport


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        transport: ToolTransport,
        max_attempts: int = 3,
    ) -> None:
        self._registry = registry
        self._transport = transport
        self._max_attempts = max_attempts

    def execute_call(self, call: ToolCall) -> ToolResult:
        """Always returns a ToolResult -- success or a structured error."""
        try:
            args = self._registry.validate_call(call)
        except ToolFailure as exc:
            # The model's own mistake: no point retrying the transport.
            return ToolResult(
                call_id=call.id,
                name=call.name,
                output=f"ERROR: {exc.reason}",
                is_error=True,
            )

        tool = self._registry.get(call.name)
        last_reason = ""
        for _attempt in range(1, self._max_attempts + 1):
            try:
                raw = self._transport.execute(call, args)
                output = self._check_observation(tool, raw)
                return ToolResult(call_id=call.id, name=call.name, output=output)
            except (TransportError, GateFailure) as exc:
                last_reason = str(exc)
                continue

        return ToolResult(
            call_id=call.id,
            name=call.name,
            output=f"ERROR after {self._max_attempts} attempts: {last_reason}",
            is_error=True,
        )

    def _check_observation(self, tool: Tool, raw: object) -> str:
        """The observation gate: only well-formed output reaches the agent."""
        if tool.output_model is None:
            return raw if isinstance(raw, str) else json.dumps(raw)

        where = f"observation:{tool.name}"
        data = raw
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise GateFailure(where, f"output is not valid JSON: {exc}") from exc
        try:
            validated = tool.output_model.model_validate(data)
        except ValidationError as exc:
            raise GateFailure(where, f"output does not match schema: {exc}") from exc
        return validated.model_dump_json()
