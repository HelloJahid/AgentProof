"""The tool execution step: drains the model's wishes into observations.

Sits between two model calls in the ReAct cycle. Everything hard about
touching the world (gates, retries, structured errors) lives in the
ToolExecutor -- this step's only job is to move data through the state:
pending_tool_calls out, tool_results and tool messages in.
"""

from agentproof.state import AgentState, Message
from agentproof.tools.executor import ToolExecutor


class ToolExecStep:
    name = "tools"

    def __init__(self, executor: ToolExecutor) -> None:
        self._executor = executor

    def run(self, state: AgentState) -> AgentState:
        while state.pending_tool_calls:
            call = state.pending_tool_calls.pop(0)
            result = self._executor.execute_call(call)
            state.tool_results.append(result)
            # The observation also enters the conversation -- linked to the
            # call it answers -- so the next model call can reason over it.
            state.messages.append(Message(role="tool", content=result.output, tool_call_id=call.id))
        return state
