"""The trajectory viewer: a trace file, readable by humans.

Two renderings of the same RunReplay: a compact terminal view for quick
inspection, and a self-contained HTML report (inline CSS, zero external
assets) you can attach to a bug report or open years later. Both consume
the replay -- never the live process -- so anything that was ever recorded
can be viewed, including runs that crashed.

The view shows each step as what it CONTRIBUTED: the messages it added and
the tool results it produced, computed by diffing consecutive state
snapshots. The full snapshots stay in the trace for machines; humans get
the deltas.
"""

import html
from pathlib import Path

from agentproof.state import Message, ToolResult
from agentproof.trace.replay import RunReplay, load_trace

_SNIP = 160


def _snip(text: str, limit: int = _SNIP) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _step_deltas(replay: RunReplay) -> list[tuple[str, float, list[Message], list[ToolResult]]]:
    """(step name, duration, new messages, new tool results) per step."""
    deltas = []
    previous_messages = 0
    previous_results = 0
    for step in replay.steps:
        deltas.append(
            (
                step.name,
                step.duration_ms,
                step.state.messages[previous_messages:],
                step.state.tool_results[previous_results:],
            )
        )
        previous_messages = len(step.state.messages)
        previous_results = len(step.state.tool_results)
    return deltas


def _describe_message(message: Message) -> str:
    text = _snip(message.content) if message.content else "(no text)"
    if message.tool_calls:
        wishes = ", ".join(
            f"{call.name}({_snip(str(call.arguments), 60)})" for call in message.tool_calls
        )
        return f"{message.role}: {text}  [requests: {wishes}]"
    return f"{message.role}: {text}"


def render_text(replay: RunReplay) -> str:
    final = replay.final_state
    tokens = f"{final.input_tokens} in / {final.output_tokens} out" if final else "n/a"

    lines = [
        f"=== run {replay.run_id} -- {replay.outcome} ===",
        f"query:  {replay.query}",
        f"path:   {' -> '.join(replay.path) or '(no steps)'}",
        f"tokens: {tokens}",
        "",
    ]
    for index, (name, duration_ms, messages, results) in enumerate(_step_deltas(replay), 1):
        lines.append(f"[{index}] {name}  ({duration_ms:.1f}ms)")
        for message in messages:
            lines.append(f"    + {_describe_message(message)}")
        for result in results:
            marker = "ERROR" if result.is_error else "ok"
            lines.append(f"    = {result.name} [{marker}]: {_snip(result.output)}")
    lines.append("")
    if replay.outcome == "failed":
        lines.append(f"FAILED: {replay.error_type}: {replay.error_message}")
    elif replay.outcome == "truncated":
        lines.append("TRUNCATED: the trace ends mid-run (process died?)")
    else:
        lines.append(f"final answer: {replay.final_answer}")
    return "\n".join(lines)


_HTML_STYLE = """
body { font-family: ui-monospace, Consolas, monospace; margin: 2rem auto;
       max-width: 60rem; background: #14161a; color: #d8dee9; }
h1 { font-size: 1.1rem; } .meta { color: #8892a0; }
details { border: 1px solid #2b313b; border-radius: 6px; margin: .5rem 0;
          padding: .4rem .8rem; background: #1b1f26; }
summary { cursor: pointer; font-weight: bold; }
.msg { margin: .3rem 0 .3rem 1rem; white-space: pre-wrap; }
.role { color: #7aa2f7; } .tool-ok { color: #9ece6a; } .tool-err { color: #f7768e; }
.answer { border-left: 3px solid #9ece6a; padding-left: .8rem; margin-top: 1rem;
          white-space: pre-wrap; }
.failed { border-left: 3px solid #f7768e; padding-left: .8rem; margin-top: 1rem; }
"""


def render_html(replay: RunReplay) -> str:
    esc = html.escape
    final = replay.final_state
    tokens = f"{final.input_tokens} in / {final.output_tokens} out" if final else "n/a"

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>AgentProof trace {esc(replay.run_id)}</title>",
        f"<style>{_HTML_STYLE}</style></head><body>",
        f"<h1>AgentProof run {esc(replay.run_id)} &mdash; {esc(replay.outcome)}</h1>",
        f"<p class='meta'>query: {esc(replay.query)}<br>",
        f"path: {esc(' -> '.join(replay.path))}<br>tokens: {esc(tokens)}</p>",
    ]

    for index, (name, duration_ms, messages, results) in enumerate(_step_deltas(replay), 1):
        parts.append(
            f"<details open><summary>[{index}] {esc(name)} ({duration_ms:.1f}ms)</summary>"
        )
        for message in messages:
            parts.append(
                f"<div class='msg'><span class='role'>{esc(message.role)}</span>: "
                f"{esc(message.content) or '(no text)'}"
                + (
                    " <em>[requests: "
                    + esc(
                        ", ".join(f"{call.name}({call.arguments})" for call in message.tool_calls)
                    )
                    + "]</em>"
                    if message.tool_calls
                    else ""
                )
                + "</div>"
            )
        for result in results:
            css = "tool-err" if result.is_error else "tool-ok"
            label = "ERROR" if result.is_error else "ok"
            parts.append(
                f"<div class='msg {css}'>{esc(result.name)} [{label}]: {esc(result.output)}</div>"
            )
        parts.append("</details>")

    if replay.outcome == "failed":
        parts.append(
            f"<div class='failed'>FAILED &mdash; {esc(replay.error_type or '')}: "
            f"{esc(replay.error_message or '')}</div>"
        )
    elif replay.outcome == "truncated":
        parts.append("<div class='failed'>TRUNCATED &mdash; the trace ends mid-run</div>")
    else:
        parts.append(f"<div class='answer'>{esc(replay.final_answer or '')}</div>")

    parts.append("</body></html>")
    return "".join(parts)


def write_html(trace_path: Path | str, out_path: Path | str) -> Path:
    replay = load_trace(trace_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(replay), encoding="utf-8")
    return out_path
