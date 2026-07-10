"""Agent state: the typed working memory carried through every step of a run.

LLMs are stateless -- each API call starts from a blank slate. The AgentState
object is the bridge: it accumulates the query, instructions, messages, tool
activity, and intermediate results for the duration of ONE task, then clears.
Every step in the state machine receives this object and returns an updated
version of it, so knowledge gathered early in the run stays visible later.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """The model's stated intent to use a tool -- not yet executed."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The observation produced by executing one ToolCall."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    name: str
    output: str
    is_error: bool = False


class Message(BaseModel):
    """One turn in the conversation the agent is building with the model.

    Assistant turns may carry the tool calls they requested; tool turns carry
    the id of the call they answer -- so the conversation alone preserves the
    full intent -> observation linkage any provider needs to see.
    """

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)  # assistant turns
    tool_call_id: str | None = None  # tool turns: which call this answers


class StepRecord(BaseModel):
    """One entry in the run's step history: which step ran and what it did."""

    model_config = ConfigDict(extra="forbid")

    step: str
    detail: str = ""


class AgentState(BaseModel):
    """The single state object every step reads from and writes back to.

    Ephemeral by design: this is working memory for one task, not durable
    memory across sessions (that arrives later as a separate concern).
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    instructions: str = ""
    messages: list[Message] = Field(default_factory=list)
    pending_tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    history: list[StepRecord] = Field(default_factory=list)
    final_answer: str | None = None
    step_count: int = 0
    # Running token totals across every model call in the run -- the raw
    # material of the system-metrics eval dimension.
    input_tokens: int = 0
    output_tokens: int = 0

    def add_message(self, role: Role, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def record_step(self, step: str, detail: str = "") -> None:
        self.history.append(StepRecord(step=step, detail=detail))
        self.step_count += 1

    @property
    def is_done(self) -> bool:
        return self.final_answer is not None
