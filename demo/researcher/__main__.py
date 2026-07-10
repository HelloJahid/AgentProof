"""CLI: ask the grounded researcher a question, live.

    python -m demo.researcher "What did NVIDIA announce at its last GTC?"

Needs ANTHROPIC_API_KEY and TAVILY_API_KEY (a .env file in the project root
is loaded automatically). Every run writes a flight-recorder trace under
runs/ -- the same file the eval harness and viewer consume.
"""

import os
import sys
import time
from pathlib import Path

from agentproof.llm import AnthropicClient
from agentproof.tools import TavilySearchTransport
from agentproof.trace import TraceRecorder
from demo.researcher.agent import build_researcher, initial_state


def load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny .env loader: KEY=VALUE lines into the environment (no override)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> int:
    load_dotenv()

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('usage: python -m demo.researcher "your question"')
        return 2
    for key in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY"):
        if not os.environ.get(key):
            print(f"missing {key} (put it in .env or the environment)")
            return 2

    machine = build_researcher(
        AnthropicClient(max_tokens=2048),
        TavilySearchTransport(os.environ["TAVILY_API_KEY"]),
    )

    trace_path = Path("runs") / f"research-{int(time.time())}.trace.jsonl"
    with TraceRecorder(trace_path) as recorder:
        state = machine.run(initial_state(question), recorder=recorder)

    print(state.final_answer or "(no answer)")
    print()
    print(f"--- path: {' -> '.join(record.step for record in state.history)}")
    print(f"--- tokens: {state.input_tokens} in / {state.output_tokens} out")
    print(f"--- trace: {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
