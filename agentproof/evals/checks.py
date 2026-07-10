"""Rule-based evaluators: cheap, deterministic, and first in line.

Rules before judges: anything a regex or a list comparison can decide should
never cost a model call. Each check judges ONE thing along ONE dimension and
says why. When a case lacks the reference a check needs, the check reports
itself not applicable instead of guessing.
"""

import re

from agentproof.evals.datasets import EvalCase
from agentproof.evals.results import CheckResult
from agentproof.trace.replay import RunReplay


class RunSucceeded:
    """Task completion, lowest bar: the run finished and produced an answer."""

    name = "run_succeeded"

    def evaluate(self, case: EvalCase, replay: RunReplay) -> CheckResult:
        if replay.outcome != "finished":
            detail = replay.error_message or "no final answer"
            return CheckResult(
                check=self.name,
                dimension="task_completion",
                passed=False,
                reason=f"run outcome was {replay.outcome!r}: {detail}",
            )
        return CheckResult(
            check=self.name,
            dimension="task_completion",
            passed=True,
            reason=f"run finished with an answer in {len(replay.steps)} steps",
        )


class AnswerMatchesReference:
    """Final-response lens: judge only the answer, against the reference."""

    name = "answer_matches_reference"

    def evaluate(self, case: EvalCase, replay: RunReplay) -> CheckResult:
        if case.reference_answer is None and case.answer_pattern is None:
            return CheckResult(
                check=self.name,
                dimension="quality",
                passed=True,
                applicable=False,
                reason="case carries no reference answer or pattern",
            )
        answer = replay.final_answer or ""

        if case.reference_answer is not None:
            passed = answer.strip() == case.reference_answer.strip()
            reason = (
                "answer matches the reference exactly"
                if passed
                else f"expected {case.reference_answer!r}, got {answer!r}"
            )
            return CheckResult(check=self.name, dimension="quality", passed=passed, reason=reason)

        assert case.answer_pattern is not None
        passed = re.search(case.answer_pattern, answer) is not None
        reason = (
            f"answer matches pattern /{case.answer_pattern}/"
            if passed
            else f"answer {answer!r} does not match pattern /{case.answer_pattern}/"
        )
        return CheckResult(check=self.name, dimension="quality", passed=passed, reason=reason)


class TookExpectedPath:
    """Trajectory lens, exact form: the run took precisely the expected route."""

    name = "took_expected_path"

    def evaluate(self, case: EvalCase, replay: RunReplay) -> CheckResult:
        if case.expected_path is None:
            return CheckResult(
                check=self.name,
                dimension="tool_interaction",
                passed=True,
                applicable=False,
                reason="case carries no expected path",
            )
        passed = replay.path == case.expected_path
        reason = (
            f"trajectory {replay.path} as expected"
            if passed
            else f"expected {case.expected_path}, took {replay.path}"
        )
        return CheckResult(
            check=self.name, dimension="tool_interaction", passed=passed, reason=reason
        )
