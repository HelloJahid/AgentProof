"""The researcher's eval suite -- the command CI runs, and you can too.

    python -m demo.researcher.evalsuite                 # mocked: the CI gate
    python -m demo.researcher.evalsuite --live          # live model + judge

Mocked mode is fully deterministic (golden scripts, rule checks only) and
guards the runtime machinery on every push. Live mode runs the same dataset
through the real model and web search, and seats the LLM-as-judge on the
panel -- the offline, sampled setting where a judge belongs.
"""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from agentproof.evals import (
    AnswerMatchesReference,
    EvalCase,
    Evaluator,
    JudgeEvaluator,
    PathIncludesSubsequence,
    RunSucceeded,
    UsedExpectedTools,
    WithinBudgets,
    load_dataset,
    run_suite,
)
from agentproof.evals.ci_gate import run_gate
from agentproof.llm import AnthropicClient, MockModelClient
from agentproof.machine import StateMachine
from agentproof.tools import MockTransport, TavilySearchTransport
from demo.researcher import golden
from demo.researcher.agent import build_researcher

RULE_CHECKS: list[Evaluator] = [
    RunSucceeded(),
    AnswerMatchesReference(),
    PathIncludesSubsequence(),
    UsedExpectedTools(),
    WithinBudgets(),
]


def mocked_factory(case: EvalCase) -> StateMachine:
    responses, transport_script = golden.scripts_for(case.id)
    return build_researcher(MockModelClient(responses), MockTransport(transport_script))


def live_factory(case: EvalCase) -> StateMachine:
    return build_researcher(
        AnthropicClient(max_tokens=2048),
        TavilySearchTransport(os.environ["TAVILY_API_KEY"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the researcher eval suite.")
    parser.add_argument("--live", action="store_true", help="real model, search, and judge")
    parser.add_argument("--dataset", default="datasets/researcher.jsonl")
    parser.add_argument("--trace-dir", default="runs/evals")
    parser.add_argument("--scorecard", default=None, help="where to save the scorecard JSON")
    args = parser.parse_args(argv)

    cases = load_dataset(args.dataset)
    evaluators = list(RULE_CHECKS)

    if args.live:
        from demo.researcher.__main__ import load_dotenv

        load_dotenv()
        for key in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY"):
            if not os.environ.get(key):
                print(f"missing {key} for --live mode")
                return 2
        factory = live_factory
        evaluators.append(JudgeEvaluator(AnthropicClient(max_tokens=512)))
    else:
        factory = mocked_factory

    suite = run_suite(cases, factory, evaluators, trace_dir=Path(args.trace_dir))
    return run_gate(suite, scorecard_path=args.scorecard)


if __name__ == "__main__":
    raise SystemExit(main())
