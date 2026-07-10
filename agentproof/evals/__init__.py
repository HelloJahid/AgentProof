"""The evaluation harness: repeatable measurement of agent behavior.

First-class deliverable, not a test folder: datasets define what to ask and
what to expect, evaluators judge replays against them, and (soon) scorecards
aggregate the verdicts into something a CI gate can act on.
"""

from agentproof.evals.checks import AnswerMatchesReference, RunSucceeded, TookExpectedPath
from agentproof.evals.datasets import EvalCase, load_dataset
from agentproof.evals.results import CheckResult, Dimension, Evaluator

__all__ = [
    "AnswerMatchesReference",
    "CheckResult",
    "Dimension",
    "EvalCase",
    "Evaluator",
    "RunSucceeded",
    "TookExpectedPath",
    "load_dataset",
]
