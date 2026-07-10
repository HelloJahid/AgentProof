"""The CI regression gate: the eval suite as a merge blocker.

Takes a judged suite, prints the scorecard where CI logs will show it,
saves the scorecard as JSON (an artifact future runs can be compared
against), and converts the verdict into the only language CI speaks:
an exit code. Green means every golden case still passes; red names the
cases that broke and why -- in the log, before anyone has to dig.
"""

from pathlib import Path

from agentproof.evals.harness import SuiteResult
from agentproof.evals.scorecard import render_scorecard


def run_gate(suite: SuiteResult, scorecard_path: Path | str | None = None) -> int:
    print(render_scorecard(suite))

    if scorecard_path is not None:
        scorecard_path = Path(scorecard_path)
        scorecard_path.parent.mkdir(parents=True, exist_ok=True)
        scorecard_path.write_text(suite.model_dump_json(indent=2), encoding="utf-8")
        print(f"scorecard saved: {scorecard_path}")

    if suite.all_passed:
        print("eval gate: PASS")
        return 0

    failed = [result.case_id for result in suite.results if not result.passed]
    print(f"eval gate: FAIL -- broken cases: {', '.join(failed)}")
    return 1
