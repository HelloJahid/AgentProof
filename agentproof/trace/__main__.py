"""CLI: view any trace file.

python -m agentproof.trace runs/research-1234.trace.jsonl
python -m agentproof.trace runs/research-1234.trace.jsonl --html report.html
"""

import argparse
from collections.abc import Sequence

from agentproof.trace.replay import load_trace
from agentproof.trace.viewer import render_text, write_html


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="View an AgentProof trace file.")
    parser.add_argument("trace", help="path to a .trace.jsonl file")
    parser.add_argument("--html", default=None, help="also write an HTML report here")
    args = parser.parse_args(argv)

    print(render_text(load_trace(args.trace)))
    if args.html:
        out = write_html(args.trace, args.html)
        print(f"\nhtml report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
