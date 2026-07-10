"""Phase 6: the LLM-as-judge -- rubric in, gate-checked verdict out."""

import pytest

from agentproof.errors import EvalFailure
from agentproof.evals import EvalCase, JudgeEvaluator
from agentproof.llm import MockModelClient, ModelResponse
from agentproof.state import AgentState, ToolResult
from agentproof.trace.replay import ReplayStep, RunReplay

CASE = EvalCase(id="gtc", query="What did NVIDIA announce at GTC 2025?")


def make_replay(answer: str | None) -> RunReplay:
    final = AgentState(query=CASE.query)
    final.tool_results.append(
        ToolResult(
            call_id="s1",
            name="web_search",
            output='{"results": [{"title": "GTC", "url": "https://techcrunch.com/gtc",'
            ' "content": "NVIDIA unveiled Blackwell Ultra."}]}',
        )
    )
    return RunReplay(
        run_id="r1",
        query=CASE.query,
        instructions="",
        steps=[ReplayStep(name="model", duration_ms=1.0, state=final)],
        outcome="finished",
        final_answer=answer,
    )


def test_judge_returns_a_typed_verdict_as_a_check_result() -> None:
    judge_model = MockModelClient(
        [ModelResponse(text='{"passed": true, "score": 0.9, "reason": "All claims cited."}')]
    )
    judge = JudgeEvaluator(judge_model)

    result = judge.evaluate(CASE, make_replay("Blackwell Ultra [https://techcrunch.com/gtc]"))

    assert result.passed and result.score == 0.9
    assert result.dimension == "quality"
    assert result.reason == "All claims cited."


def test_judge_sees_rubric_question_evidence_and_answer() -> None:
    judge_model = MockModelClient(
        [ModelResponse(text='{"passed": false, "score": 0.2, "reason": "Uncited claim."}')]
    )
    judge = JudgeEvaluator(judge_model)

    judge.evaluate(CASE, make_replay("NVIDIA announced something big."))

    prompt = judge_model.calls[0]["messages"][0].content
    assert "RUBRIC:" in prompt and "cites a source URL" in prompt
    assert "QUESTION:\nWhat did NVIDIA announce at GTC 2025?" in prompt
    assert "EVIDENCE" in prompt and "Blackwell Ultra" in prompt  # the observations
    assert "ANSWER" in prompt and "something big" in prompt
    # And the anti-bias instruction rode along as the system prompt:
    assert "Do not reward length" in judge_model.calls[0]["instructions"]


def test_judge_tolerates_json_wrapped_in_prose_or_fences() -> None:
    judge_model = MockModelClient(
        [
            ModelResponse(
                text='Here is my verdict:\n```json\n{"passed": true, "score": 1.0, '
                '"reason": "Grounded."}\n```'
            )
        ]
    )

    result = JudgeEvaluator(judge_model).evaluate(CASE, make_replay("answer"))

    assert result.passed and result.score == 1.0


def test_invalid_verdict_triggers_retry_with_feedback() -> None:
    judge_model = MockModelClient(
        [
            ModelResponse(text="It looks fine to me!"),  # not a verdict
            ModelResponse(text='{"passed": true, "score": 0.8, "reason": "OK."}'),
        ]
    )

    result = JudgeEvaluator(judge_model).evaluate(CASE, make_replay("answer"))

    assert result.passed
    # The second call carried the judge's bad reply AND what was wrong with it:
    retry_messages = judge_model.calls[1]["messages"]
    assert retry_messages[-2].content == "It looks fine to me!"
    assert "not a valid verdict" in retry_messages[-1].content


def test_a_judge_that_never_complies_is_a_broken_instrument_not_a_grade() -> None:
    judge_model = MockModelClient([ModelResponse(text="nope"), ModelResponse(text="still nope")])

    with pytest.raises(EvalFailure, match="no valid verdict"):
        JudgeEvaluator(judge_model, max_attempts=2).evaluate(CASE, make_replay("answer"))


def test_out_of_range_score_is_rejected_and_retried() -> None:
    judge_model = MockModelClient(
        [
            ModelResponse(text='{"passed": true, "score": 7, "reason": "great!"}'),
            ModelResponse(text='{"passed": true, "score": 0.7, "reason": "great."}'),
        ]
    )

    result = JudgeEvaluator(judge_model).evaluate(CASE, make_replay("answer"))

    assert result.score == 0.7  # the 7 never left the gate
