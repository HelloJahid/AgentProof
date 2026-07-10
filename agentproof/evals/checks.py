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


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """True when needle's items appear in haystack, in order, gaps allowed."""
    position = 0
    for item in haystack:
        if position < len(needle) and item == needle[position]:
            position += 1
    return position == len(needle)


class PathIncludesSubsequence:
    """Trajectory lens, tolerant form: the expected route appears IN ORDER
    within the actual path, extra steps allowed.

    Built for live models: a real LLM may search twice or reason an extra
    round, and that is not a failure -- as long as the essential shape
    (e.g. model -> tools -> model) is present. Exact matching is for
    deterministic mocks; this is for the world.
    """

    name = "path_includes_expected_subsequence"

    def evaluate(self, case: EvalCase, replay: RunReplay) -> CheckResult:
        if case.expected_path is None:
            return CheckResult(
                check=self.name,
                dimension="tool_interaction",
                passed=True,
                applicable=False,
                reason="case carries no expected path",
            )
        passed = _is_subsequence(case.expected_path, replay.path)
        reason = (
            f"path {replay.path} contains {case.expected_path} in order"
            if passed
            else f"path {replay.path} is missing {case.expected_path} (in order)"
        )
        return CheckResult(
            check=self.name, dimension="tool_interaction", passed=passed, reason=reason
        )


class UsedExpectedTools:
    """Tool interaction: every expected tool was actually called (any order)."""

    name = "used_expected_tools"

    def evaluate(self, case: EvalCase, replay: RunReplay) -> CheckResult:
        if case.expected_tools is None:
            return CheckResult(
                check=self.name,
                dimension="tool_interaction",
                passed=True,
                applicable=False,
                reason="case carries no expected tools",
            )
        final = replay.final_state
        called = [result.name for result in final.tool_results] if final else []
        missing = [tool for tool in case.expected_tools if tool not in called]
        passed = not missing
        reason = (
            f"all expected tools were used: {case.expected_tools}"
            if passed
            else f"missing tool calls: {missing} (called: {called or 'none'})"
        )
        return CheckResult(
            check=self.name, dimension="tool_interaction", passed=passed, reason=reason
        )


class WithinBudgets:
    """System metrics: the run stayed inside its step and token budgets.

    An agent that answers correctly in 40 steps and 200k tokens has still
    failed at something -- this is the dimension that notices.
    """

    name = "within_budgets"

    def evaluate(self, case: EvalCase, replay: RunReplay) -> CheckResult:
        if case.max_steps is None and case.max_total_tokens is None:
            return CheckResult(
                check=self.name,
                dimension="system",
                passed=True,
                applicable=False,
                reason="case carries no budgets",
            )
        problems: list[str] = []
        steps_taken = len(replay.steps)
        if case.max_steps is not None and steps_taken > case.max_steps:
            problems.append(f"{steps_taken} steps > budget {case.max_steps}")
        final = replay.final_state
        tokens = (final.input_tokens + final.output_tokens) if final else 0
        if case.max_total_tokens is not None and tokens > case.max_total_tokens:
            problems.append(f"{tokens} tokens > budget {case.max_total_tokens}")
        passed = not problems
        reason = (
            f"within budgets: {steps_taken} steps, {tokens} tokens"
            if passed
            else "; ".join(problems)
        )
        return CheckResult(check=self.name, dimension="system", passed=passed, reason=reason)


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
