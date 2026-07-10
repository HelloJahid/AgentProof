"""The grounded web researcher, assembled from runtime parts.

Nothing here is new machinery -- that is the point of the demo. A tool
declaration, an instruction block, and the same machine/steps/router the
tests have exercised all along. The only choice that makes it a RESEARCHER
is the instructions: answers must be grounded, every claim tied to a source
the reader can follow, and honesty required when the evidence is not there.
"""

from pydantic import BaseModel, Field

from agentproof.llm import ModelClient
from agentproof.machine import StateMachine
from agentproof.memory import SlidingWindow
from agentproof.state import AgentState
from agentproof.steps import ModelCallStep, PrepareStep, ToolExecStep, react_router
from agentproof.tools import Tool, ToolExecutor, ToolRegistry, ToolTransport

INSTRUCTIONS = (
    "You are a careful research agent. Use the web_search tool to find "
    "current information before answering.\n"
    "Rules:\n"
    "1. Ground every claim: each factual statement must cite its source URL "
    "in square brackets immediately after the claim, like [https://example.com].\n"
    "2. Prefer recent, trustworthy sources; ignore clickbait.\n"
    "3. If the search results do not answer the question, say plainly that "
    "you could not find reliable information -- never invent an answer.\n"
    "4. Keep the final answer short: a few sentences, then a Sources list."
)


class SearchArgs(BaseModel):
    query: str = Field(min_length=3, description="The web search query.")
    max_results: int = Field(default=5, ge=1, le=10)


class SearchResult(BaseModel):
    title: str
    url: str
    content: str


class SearchResults(BaseModel):
    results: list[SearchResult]


web_search = Tool(
    name="web_search",
    description=(
        "Search the web for current information. Returns a list of results "
        "with title, url, and a content snippet."
    ),
    input_model=SearchArgs,
    output_model=SearchResults,
)


def build_researcher(
    model_client: ModelClient,
    search_transport: ToolTransport,
    max_steps: int = 12,
) -> StateMachine:
    registry = ToolRegistry([web_search])
    return StateMachine(
        [
            PrepareStep(),
            ModelCallStep(model_client, tools=registry.specs(), memory=SlidingWindow(20)),
            ToolExecStep(ToolExecutor(registry, search_transport)),
        ],
        router=react_router,
        start="prepare",
        max_steps=max_steps,
    )


def initial_state(question: str) -> AgentState:
    return AgentState(query=question, instructions=INSTRUCTIONS)
