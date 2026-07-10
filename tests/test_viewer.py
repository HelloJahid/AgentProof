"""Phase 7: any trace file becomes a human-readable story."""

from pathlib import Path

import pytest

from agentproof.errors import MaxStepsExceeded
from agentproof.llm import MockModelClient
from agentproof.machine import StateMachine
from agentproof.state import AgentState
from agentproof.tools import MockTransport
from agentproof.trace import TraceRecorder, load_trace, render_html, render_text
from agentproof.trace.__main__ import main as viewer_cli
from demo.researcher import build_researcher, initial_state
from demo.researcher.golden import scripts_for


def record_researcher_run(tmp_path: Path, question: str = "What did NVIDIA announce?") -> Path:
    responses, script = scripts_for("gtc-announcement")
    machine = build_researcher(MockModelClient(responses), MockTransport(script))
    trace_path = tmp_path / "run.trace.jsonl"
    with TraceRecorder(trace_path) as recorder:
        machine.run(initial_state(question), recorder=recorder)
    return trace_path


def test_text_view_tells_the_whole_story(tmp_path: Path) -> None:
    replay = load_trace(record_researcher_run(tmp_path))

    view = render_text(replay)

    assert "-- finished ===" in view
    assert "path:   prepare -> model -> tools -> model" in view
    assert "tokens: 1000 in / 130 out" in view
    assert "[requests: web_search(" in view  # the model's wish, visible
    assert "= web_search [ok]:" in view  # the observation that answered it
    assert "final answer: NVIDIA unveiled its Blackwell Ultra chip" in view


def test_text_view_shows_failures_and_truncation(tmp_path: Path) -> None:
    class Greet:
        name = "greet"

        def run(self, state: AgentState) -> AgentState:
            state.add_message("assistant", "hi")
            return state

    trace_path = tmp_path / "crash.trace.jsonl"
    machine = StateMachine([Greet()], router=lambda s, c: "greet", max_steps=2)
    with TraceRecorder(trace_path) as recorder:
        with pytest.raises(MaxStepsExceeded):
            machine.run(AgentState(query="loop"), recorder=recorder)

    view = render_text(load_trace(trace_path))
    assert "FAILED: MaxStepsExceeded" in view

    # Chop the tail off: now it is a truncated story.
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    trace_path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    assert "TRUNCATED" in render_text(load_trace(trace_path))


def test_html_report_is_self_contained_and_escaped(tmp_path: Path) -> None:
    trace_path = record_researcher_run(
        tmp_path, question="<script>alert('xss')</script> what happened?"
    )

    html_report = render_html(load_trace(trace_path))

    assert html_report.startswith("<!doctype html>")
    assert "<script>alert" not in html_report  # hostile input neutralised
    assert "&lt;script&gt;" in html_report
    assert "http" not in html_report.split("<style>")[1].split("</style>")[0]  # no external CSS
    assert "tool-ok" in html_report
    assert "Blackwell Ultra" in html_report


def test_cli_prints_the_view_and_writes_the_report(tmp_path: Path, capsys) -> None:
    trace_path = record_researcher_run(tmp_path)
    out_path = tmp_path / "report.html"

    exit_code = viewer_cli([str(trace_path), "--html", str(out_path)])

    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "path:   prepare -> model -> tools -> model" in printed
    assert out_path.exists()
    assert "AgentProof run" in out_path.read_text(encoding="utf-8")
