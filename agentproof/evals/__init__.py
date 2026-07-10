"""The evaluation harness: repeatable measurement of agent behavior.

First-class deliverable, not a test folder: datasets define what to ask and
what to expect, evaluators judge replays against them, and (soon) scorecards
aggregate the verdicts into something a CI gate can act on.
"""

from agentproof.evals.checks import (
    AnswerMatchesReference,
    PathIncludesSubsequence,
    RunSucceeded,
    TookExpectedPath,
    UsedExpectedTools,
    WithinBudgets,
)
from agentproof.evals.datasets import EvalCase, load_dataset
from agentproof.evals.harness import CaseResult, SuiteResult, run_suite
from agentproof.evals.results import CheckResult, Dimension, Evaluator
from agentproof.evals.scorecard import render_scorecard

__all__ = [
    "AnswerMatchesReference",
    "CaseResult",
    "CheckResult",
    "Dimension",
    "EvalCase",
    "Evaluator",
    "PathIncludesSubsequence",
    "RunSucceeded",
    "SuiteResult",
    "TookExpectedPath",
    "UsedExpectedTools",
    "WithinBudgets",
    "load_dataset",
    "render_scorecard",
    "run_suite",
]
