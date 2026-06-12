"""Deep Researcher: an ADK-orchestrated, human-steered research agent."""

from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    """Terminal REPL — the minimal way to drive a project without the UI."""
    parser = argparse.ArgumentParser(prog="deep-researcher")
    parser.add_argument("--project", help="Project id (slug). Default: derived from question.")
    parser.add_argument("question", nargs="*", help="Initial research question.")
    args = parser.parse_args()

    from .config import get_settings
    from .runner import build_runner, event_text, event_tool_calls, run_turn, slugify

    settings = get_settings()
    question = " ".join(args.question).strip()
    project_id = args.project or (slugify(question) if question else None)
    if not project_id:
        parser.error("Provide a research question or --project to resume one.")

    runner = build_runner()
    print(f"project: {project_id}")
    print(f"artifacts: {settings.projects_dir / project_id}\n")

    async def turn(message: str) -> None:
        async for event in run_turn(runner, project_id, message):
            for tool in event_tool_calls(event):
                print(f"  [{event.author} → {tool}]")
            text = event_text(event)
            if text and not event.partial:
                print(f"\n{event.author}: {text}\n")

    pending = question
    try:
        while True:
            if pending:
                asyncio.run(turn(pending))
                pending = ""
            user_input = input("you> ").strip()
            if user_input.lower() in {"exit", "quit", "/exit", "/quit"}:
                break
            if user_input:
                pending = user_input
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
