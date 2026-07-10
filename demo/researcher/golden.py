"""Golden scripts: the deterministic twin of every golden dataset case.

Each case id maps to the scripted model responses and search payloads that a
CORRECT runtime should turn into a passing run. The mocked suite therefore
guards the machinery -- router, gates, executor, memory, state handling: if
a change anywhere in the runtime bends any golden trajectory or answer, the
gate goes red. (Prompt-quality regressions need the live model: run the same
suite with --live, where the LLM-as-judge joins the evaluator panel.)
"""

from typing import Any

from agentproof.errors import EvalFailure
from agentproof.llm import ModelResponse
from agentproof.state import ToolCall

_TECHCRUNCH = {
    "results": [
        {
            "title": "NVIDIA GTC 2025 recap",
            "url": "https://techcrunch.com/gtc-2025",
            "content": "NVIDIA unveiled its Blackwell Ultra chip at GTC 2025.",
        }
    ]
}

_SCRIPTS: dict[str, tuple[list[ModelResponse], dict[str, list[Any]]]] = {
    "gtc-announcement": (
        [
            ModelResponse(
                text="Searching for the GTC announcement.",
                tool_calls=[
                    ToolCall(
                        id="s1",
                        name="web_search",
                        arguments={"query": "NVIDIA GTC 2025 announcement"},
                    )
                ],
                input_tokens=300,
                output_tokens=40,
            ),
            ModelResponse(
                text=(
                    "NVIDIA unveiled its Blackwell Ultra chip at GTC 2025 "
                    "[https://techcrunch.com/gtc-2025].\n\nSources:\n"
                    "- https://techcrunch.com/gtc-2025"
                ),
                input_tokens=700,
                output_tokens=90,
            ),
        ],
        {"web_search": [_TECHCRUNCH]},
    ),
    "search-down": (
        [
            ModelResponse(
                text="Searching for the launch date.",
                tool_calls=[
                    ToolCall(
                        id="s1",
                        name="web_search",
                        arguments={"query": "latest SpaceX launch date"},
                    )
                ],
                input_tokens=300,
                output_tokens=40,
            ),
            ModelResponse(
                text="I could not find reliable information on this right now.",
                input_tokens=500,
                output_tokens=40,
            ),
        ],
        {"web_search": []},  # transport empty: every attempt fails the gate
    ),
    "greeting-no-search": (
        [
            ModelResponse(
                text="Hello! Ask me a research question and I will find sources.",
                input_tokens=200,
                output_tokens=30,
            )
        ],
        {},
    ),
}


def scripts_for(case_id: str) -> tuple[list[ModelResponse], dict[str, list[Any]]]:
    """Fresh copies every call -- one run must never consume another's script."""
    if case_id not in _SCRIPTS:
        raise EvalFailure(
            f"golden case {case_id!r} has no script -- add one to demo/researcher/golden.py"
        )
    responses, transport_script = _SCRIPTS[case_id]
    return (
        [response.model_copy(deep=True) for response in responses],
        {name: list(queue) for name, queue in transport_script.items()},
    )
