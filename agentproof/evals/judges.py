"""LLM-as-judge: a model grading what rules cannot reach.

A regex can prove an answer CONTAINS a URL; it cannot prove the claims are
actually SUPPORTED by the evidence behind that URL. For that, we put a model
in the evaluator seat -- carefully, because a judge model carries known
biases (toward length, position, eloquence, and its own style) and is not
deterministic. The mitigations here come straight from the notes:

  * a FIXED RUBRIC -- the judge applies written criteria, never taste;
  * a STRUCTURED VERDICT -- typed JSON, gate-checked with retry-with-feedback
    (the PromptProof pattern, now judging instead of generating);
  * EVIDENCE ON THE TABLE -- the judge sees the tool observations the agent
    saw, so "grounded" is checked against the actual evidence, not vibes;
  * OFFLINE BY DESIGN -- judges run in eval suites on samples and golden
    cases, never as a label on every live turn.

The judge speaks the same Evaluator protocol as every rule check, through
the same ModelClient port -- so in tests the judge itself is a scripted mock.
"""

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentproof.errors import EvalFailure
from agentproof.evals.datasets import EvalCase
from agentproof.evals.results import CheckResult, Dimension
from agentproof.llm import ModelClient
from agentproof.state import Message
from agentproof.trace.replay import RunReplay

GROUNDEDNESS_RUBRIC = (
    "PASS only if ALL of the following hold:\n"
    "1. Every factual claim in the answer is supported by the evidence.\n"
    "2. Every claim cites a source URL, and every cited URL appears in the "
    "evidence.\n"
    "3. Nothing material in the answer is invented beyond the evidence.\n"
    "If the answer honestly states that no reliable information was found, "
    "PASS only when the evidence indeed contains no usable information."
)

_JUDGE_INSTRUCTIONS = (
    "You are a strict evaluation judge. Grade the ANSWER against the RUBRIC "
    "using only the EVIDENCE provided.\n"
    "Do not reward length, style, or confidence. Short and correct beats "
    "long and impressive.\n"
    'Reply with ONLY a JSON object: {"passed": true|false, '
    '"score": 0.0-1.0, "reason": "one or two sentences"}.'
)

_MAX_EVIDENCE_CHARS = 1500  # per observation, to keep the judge prompt bounded


class JudgeVerdict(BaseModel):
    # Lenient on extras (an external model may add fields; harmless additions
    # must not break the measurement) -- strict on the essentials.
    model_config = ConfigDict(extra="ignore")

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str


def _extract_json(text: str) -> str:
    """Tolerate a judge that wraps its JSON in prose or code fences."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return text
    return text[start : end + 1]


class JudgeEvaluator:
    """An Evaluator whose verdict comes from a model applying a fixed rubric."""

    def __init__(
        self,
        client: ModelClient,
        rubric: str = GROUNDEDNESS_RUBRIC,
        name: str = "llm_judge_groundedness",
        dimension: Dimension = "quality",
        max_attempts: int = 2,
    ) -> None:
        self.name = name
        self._client = client
        self._rubric = rubric
        self._dimension: Dimension = dimension
        self._max_attempts = max_attempts

    def evaluate(self, case: EvalCase, replay: RunReplay) -> CheckResult:
        prompt = self._build_prompt(case, replay)
        messages = [Message(role="user", content=prompt)]

        last_error = ""
        for _attempt in range(1, self._max_attempts + 1):
            response = self._client.complete(_JUDGE_INSTRUCTIONS, messages)
            raw = response.text or ""
            try:
                verdict = JudgeVerdict.model_validate_json(_extract_json(raw))
            except ValidationError as exc:
                last_error = str(exc)
                # Gate-checked retry WITH FEEDBACK: tell the judge what was
                # wrong with its reply, so the next attempt can fix it.
                messages = messages + [
                    Message(role="assistant", content=raw),
                    Message(
                        role="user",
                        content=(
                            f"Your reply was not a valid verdict: {exc}. "
                            "Reply with ONLY the JSON object."
                        ),
                    ),
                ]
                continue
            return CheckResult(
                check=self.name,
                dimension=self._dimension,
                passed=verdict.passed,
                score=verdict.score,
                reason=verdict.reason,
            )

        # The MEASUREMENT failed, not the agent: a broken measuring stick
        # must never be read as a grade.
        raise EvalFailure(
            f"judge {self.name!r} produced no valid verdict "
            f"after {self._max_attempts} attempts: {last_error}"
        )

    def _build_prompt(self, case: EvalCase, replay: RunReplay) -> str:
        final = replay.final_state
        observations = final.tool_results if final else []
        if observations:
            evidence = "\n\n".join(
                f"[observation {i} | tool={result.name} | error={result.is_error}]\n"
                f"{result.output[:_MAX_EVIDENCE_CHARS]}"
                for i, result in enumerate(observations, start=1)
            )
        else:
            evidence = "(no tool observations were collected in this run)"

        return (
            f"RUBRIC:\n{self._rubric}\n\n"
            f"QUESTION:\n{replay.query}\n\n"
            f"EVIDENCE (everything the agent observed):\n{evidence}\n\n"
            f"ANSWER (what the agent finally said):\n"
            f"{replay.final_answer or '(no answer was produced)'}"
        )
