"""The model call step: the Thought phase of ReAct, as a machine step.

Sends the conversation through the injected ModelClient and writes whatever
comes back into the state: the assistant turn (with any tool calls it
carries), the pending tool wishes for the router to act on, the token bill,
and -- when the model is done wanting tools -- the final answer.

Note what this step does NOT know: which provider it is talking to, and how
tools get executed. It reads state, calls the port, writes state.
"""

from typing import Any

from agentproof.llm import ModelClient
from agentproof.state import AgentState, Message


class ModelCallStep:
    name = "model"

    def __init__(self, client: ModelClient, tools: list[dict[str, Any]] | None = None) -> None:
        self._client = client
        self._tools = tools or []

    def run(self, state: AgentState) -> AgentState:
        response = self._client.complete(state.instructions, state.messages, self._tools)

        state.input_tokens += response.input_tokens
        state.output_tokens += response.output_tokens

        state.messages.append(
            Message(
                role="assistant",
                content=response.text or "",
                tool_calls=response.tool_calls,
            )
        )

        if response.wants_tools:
            state.pending_tool_calls.extend(response.tool_calls)
        else:
            state.final_answer = response.text or ""
        return state
