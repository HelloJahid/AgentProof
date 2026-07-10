"""Routers: the conditional-transition policies the machine can follow."""

from agentproof.state import AgentState


def react_router(state: AgentState, current: str) -> str | None:
    """The ReAct cycle as a routing policy.

    Tools requested -> execute them; after tools -> back to the model to
    reason over the observations; otherwise stop. (The machine also stops on
    its own the moment the state carries a final answer.)
    """
    if state.pending_tool_calls:
        return "tools"
    if current == "tools":
        return "model"
    return None
