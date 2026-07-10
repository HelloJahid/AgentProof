"""The evaluator contract: every judge, rule or model, speaks this shape.

A CheckResult answers four questions at once: which check ran, did it pass,
along which DIMENSION of behavior was it judging, and -- always -- why. The
reason string is not decoration: when a score regresses in CI, the reason is
the first thing a human reads.
"""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from agentproof.evals.datasets import EvalCase
from agentproof.trace.replay import RunReplay

# The four dimensions of agent behavior a check can measure:
#   task_completion -- did it achieve the goal?
#   quality         -- did it follow format, instructions, and context?
#   tool_interaction-- did it use the right tools, correctly?
#   system          -- resources: steps, tokens, latency, silent failures.
Dimension = Literal["task_completion", "quality", "tool_interaction", "system"]


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: str  # which evaluator produced this
    dimension: Dimension
    passed: bool
    reason: str  # human-readable: WHY it passed or failed
    score: float | None = None  # optional graded score, 0.0..1.0
    applicable: bool = True  # False when the case lacks the needed reference


class Evaluator(Protocol):
    """Anything that can judge a replay against a case.

    Evaluators read evidence (the RunReplay) -- they never run the agent and
    never touch the live process. Rule-based checks implement this today; the
    LLM-as-judge implements the same protocol later. The harness cannot tell
    them apart, which is the point.
    """

    name: str

    def evaluate(self, case: EvalCase, replay: RunReplay) -> CheckResult: ...
