"""Eval datasets: the questions we ask the agent, and what we expect back.

An EvalCase is one repeatable experiment: the inputs to give the agent, and
the reference data to judge the run against. References are deliberately
optional and layered -- a perfect ground truth is not always available, so a
case may carry an exact expected answer, a regex the answer must match, an
expected trajectory, expected tools, or any subset. Each evaluator uses the
reference it needs and reports "not applicable" when a case lacks it.

A dataset is a JSONL file of cases -- the same format as traces: one typed
record per line, versionable in git, diffable in review.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from agentproof.errors import EvalFailure


class EvalCase(BaseModel):
    """One experiment: inputs for the agent, references for the judges."""

    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    instructions: str = ""

    # Reference data -- any subset may be present:
    reference_answer: str | None = None  # exact expected final answer
    answer_pattern: str | None = None  # regex the final answer must match
    expected_path: list[str] | None = None  # the trajectory the run should take
    expected_tools: list[str] | None = None  # tools that should be called

    # Budgets (system dimension) -- how much the run is ALLOWED to cost:
    max_steps: int | None = None
    max_total_tokens: int | None = None

    tags: list[str] = []


def load_dataset(path: Path | str) -> list[EvalCase]:
    """Load a JSONL dataset, failing loudly on malformed or duplicate cases.

    A broken dataset raises EvalFailure rather than skipping lines: silently
    dropping cases would shrink the measuring stick without telling anyone.
    """
    # utf-8-sig: hand-edited datasets from Windows editors often carry a BOM,
    # and a byte-order mark must not read as "invalid JSON".
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    cases: list[EvalCase] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.model_validate_json(line))
        except ValidationError as exc:
            raise EvalFailure(f"invalid case at line {lineno}: {exc}") from exc

    if not cases:
        raise EvalFailure(f"dataset {path} contains no cases")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise EvalFailure(f"duplicate case ids: {duplicates}")
    return cases
