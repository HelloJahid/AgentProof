"""The prepare step: seed the conversation from the state's raw inputs.

First step of every run: turn the bare query and instructions into the
opening turns of a conversation. Idempotent on purpose -- if messages
already exist (a resumed or multi-turn run), it adds nothing.
"""

from agentproof.state import AgentState


class PrepareStep:
    name = "prepare"

    def run(self, state: AgentState) -> AgentState:
        if state.messages:
            return state
        if state.instructions:
            state.add_message("system", state.instructions)
        state.add_message("user", state.query)
        return state
