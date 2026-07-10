"""AgentProof: an evaluation-first agent runtime with a trajectory flight recorder.

The curated public API. Deeper tools live in their subpackages:
agentproof.tools (registry, executor, transports), agentproof.trace
(recorder, replay, viewer), agentproof.evals (datasets, checks, judge,
harness, gate), agentproof.steps (prepare, model_call, tool_exec, routers).
"""

from agentproof.errors import (
    AgentProofError,
    EvalFailure,
    GateFailure,
    MaxStepsExceeded,
    ReplayError,
    ToolFailure,
    TransitionError,
    TransportError,
)
from agentproof.llm import AnthropicClient, MockModelClient, ModelClient, ModelResponse
from agentproof.machine import Router, StateMachine, Step
from agentproof.memory import FullHistory, MemoryPolicy, SlidingWindow, SummarizingMemory
from agentproof.state import AgentState, Message, ToolCall, ToolResult

__version__ = "0.1.0"

__all__ = [
    "AgentProofError",
    "AgentState",
    "AnthropicClient",
    "EvalFailure",
    "FullHistory",
    "GateFailure",
    "MaxStepsExceeded",
    "MemoryPolicy",
    "Message",
    "MockModelClient",
    "ModelClient",
    "ModelResponse",
    "ReplayError",
    "Router",
    "SlidingWindow",
    "Step",
    "StateMachine",
    "SummarizingMemory",
    "ToolCall",
    "ToolFailure",
    "ToolResult",
    "TransitionError",
    "TransportError",
    "__version__",
]
