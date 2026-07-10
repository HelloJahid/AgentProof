"""The suite runner: dataset in, judged evidence out.

For every case: build a fresh agent, run it with the flight recorder on,
load the trace back as a replay, and put that replay in front of every
evaluator. The harness never inspects the live run -- it judges only what
the trace can prove, so anything it reports can be re-verified from the
trace files it leaves behind.

A case whose run CRASHES is still evaluated: the replay exists (crash-safe
recorder), RunSucceeded fails it with the error in the reason, and the suite
carries on. One broken case must never take down the measurement of the rest.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from agentproof.errors import AgentProofError
from agentproof.evals.datasets import EvalCase
from agentproof.evals.results import CheckResult, Dimension, Evaluator
from agentproof.machine import StateMachine
from agentproof.state import AgentState
from agentproof.trace.recorder import TraceRecorder
from agentproof.trace.replay import load_trace

# A fresh machine per case: transports may be scripted/stateful, and one
# case's leftovers must never leak into the next measurement.
AgentFactory = Callable[[EvalCase], StateMachine]


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    outcome: Literal["finished", "failed", "truncated"]
    path: list[str]
    final_answer: str | None
    checks: list[CheckResult]
    passed: bool  # every applicable check passed
    trace_path: str  # the evidence this verdict can be re-verified from


class SuiteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[CaseResult]

    @property
    def all_passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.passed for result in self.results) / len(self.results)

    def by_dimension(self) -> dict[Dimension, tuple[int, int]]:
        """(passed, total) applicable checks per behavior dimension."""
        totals: dict[Dimension, tuple[int, int]] = {}
        for result in self.results:
            for check in result.checks:
                if not check.applicable:
                    continue
                passed, total = totals.get(check.dimension, (0, 0))
                totals[check.dimension] = (passed + int(check.passed), total + 1)
        return totals


def run_suite(
    cases: Sequence[EvalCase],
    agent_factory: AgentFactory,
    evaluators: Sequence[Evaluator],
    trace_dir: Path | str,
) -> SuiteResult:
    trace_dir = Path(trace_dir)
    results: list[CaseResult] = []

    for case in cases:
        trace_path = trace_dir / f"{case.id}.trace.jsonl"
        machine = agent_factory(case)
        state = AgentState(query=case.query, instructions=case.instructions)

        with TraceRecorder(trace_path) as recorder:
            try:
                machine.run(state, recorder=recorder)
            except AgentProofError:
                pass  # the recorder captured the failure; the replay will show it

        replay = load_trace(trace_path)
        checks = [evaluator.evaluate(case, replay) for evaluator in evaluators]
        results.append(
            CaseResult(
                case_id=case.id,
                outcome=replay.outcome,
                path=replay.path,
                final_answer=replay.final_answer,
                checks=checks,
                passed=all(check.passed for check in checks if check.applicable),
                trace_path=str(trace_path),
            )
        )

    return SuiteResult(results=results)
