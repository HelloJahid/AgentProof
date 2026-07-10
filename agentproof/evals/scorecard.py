"""The scorecard: a suite's verdicts, readable at a glance.

Plain text on purpose -- it renders identically in a terminal, a CI log, and
a pasted GitHub comment. Failed checks print their reasons, because a score
without a why is a number nobody can act on.
"""

from agentproof.evals.harness import SuiteResult

_MARK = {True: "PASS", False: "FAIL"}


def render_scorecard(suite: SuiteResult) -> str:
    lines: list[str] = ["=== AgentProof scorecard ===", ""]

    for result in suite.results:
        lines.append(f"[{_MARK[result.passed]}] {result.case_id}  ({result.outcome})")
        lines.append(f"       path: {' -> '.join(result.path) or '(none)'}")
        for check in result.checks:
            if not check.applicable:
                continue
            lines.append(f"       {_MARK[check.passed]:4} {check.check}: {check.reason}")
        lines.append("")

    lines.append("--- by dimension ---")
    for dimension, (passed, total) in sorted(suite.by_dimension().items()):
        lines.append(f"{dimension:16} {passed}/{total}")

    lines.append("")
    passed_count = sum(result.passed for result in suite.results)
    lines.append(
        f"overall: {passed_count}/{len(suite.results)} cases passed ({suite.pass_rate:.0%})"
    )
    return "\n".join(lines)
