"""Artifact tools for agents (design §7: reference-passing only).

Agents never paste large content into chat/state — they write artifacts and
pass ``{filename, summary}`` references. These tools wrap the ADK artifact
service (backed by LocalArtifactService) and the project conventions:
``decisions/decisions.md`` (ADR log) and ``checkpoints/`` (gate records).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from google.adk.tools import ToolContext
from google.genai import types

_READ_CAP = 24_000


async def write_artifact(
    filename: str,
    content: str,
    kind: str,
    title: str,
    summary: str,
    tool_context: ToolContext,
    append: bool = False,
) -> dict[str, Any]:
    """Save a text artifact (markdown/json) to the project's artifact store.

    Args:
        filename: Project-relative path, e.g. "brief/research_brief.md",
            "plan/plan.md", "lit/facet_1/notes.md", "reports/final_report.md".
        content: Full text content of the artifact (or the next part, with
            append=True).
        kind: One of: brief, design, plan, decision, checkpoint, lit_notes,
            hypothesis, analysis, report, other.
        title: Short human-readable title.
        summary: 1-3 sentence summary (other agents see this instead of the body).
        append: Append content to the artifact's latest version instead of
            replacing it. Use this to build long documents across several
            calls — each call's content stays small.

    Returns:
        {filename, version, summary} reference to pass around.
    """
    if append:
        prior = await tool_context.load_artifact(filename)
        if prior is not None and prior.text:
            content = prior.text + "\n" + content
    version = await tool_context.save_artifact(
        filename,
        types.Part(text=content),
        custom_metadata={
            "kind": kind,
            "title": title,
            "summary": summary,
            "created_by": tool_context.agent_name,
        },
    )
    return {"filename": filename, "version": version, "summary": summary}


async def read_artifact(
    filename: str, tool_context: ToolContext, version: Optional[int] = None
) -> dict[str, Any]:
    """Load the full text of a previously saved artifact.

    Args:
        filename: Project-relative path as returned by write_artifact/list_artifacts.
        version: Specific version; omit for the latest.

    Returns:
        {filename, content} or {error}.
    """
    part = await tool_context.load_artifact(filename, version=version)
    if part is None or part.text is None:
        return {"error": f"No text artifact found at {filename!r}"}
    text = part.text
    if len(text) > _READ_CAP:
        text = text[:_READ_CAP] + f"\n…[truncated, {len(part.text)} chars total]"
    return {"filename": filename, "content": text}


async def list_artifacts(tool_context: ToolContext) -> dict[str, Any]:
    """List all artifact filenames saved in this project."""
    return {"filenames": await tool_context.list_artifacts()}


async def append_decision(
    context: str, decision: str, evidence: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Append an ADR-style entry to the project decision log (decisions/decisions.md).

    Args:
        context: What situation prompted the decision.
        decision: What was decided.
        evidence: Supporting artifacts/observations, e.g. "plan/plan.md v2; user approval".
    """
    existing = await tool_context.load_artifact("decisions/decisions.md")
    body = existing.text if existing and existing.text else "# Decision Log\n"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body += (
        f"\n## {stamp} — {tool_context.agent_name}\n"
        f"- **Context:** {context}\n- **Decision:** {decision}\n- **Evidence:** {evidence}\n"
    )
    version = await tool_context.save_artifact(
        "decisions/decisions.md",
        types.Part(text=body),
        custom_metadata={
            "kind": "decision",
            "title": "Decision log",
            "summary": f"Latest: {decision[:140]}",
            "created_by": tool_context.agent_name,
        },
    )
    return {"filename": "decisions/decisions.md", "version": version}


async def record_checkpoint(
    gate: str, decision: str, comments: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Persist a human checkpoint decision (design §11) as a checkpoint artifact.

    Args:
        gate: Gate name, e.g. "plan_approval".
        decision: "approved" | "rejected" | "revised".
        comments: The user's free-text feedback, verbatim (empty string if none).
    """
    import json

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"checkpoints/{stamp}_{gate}.json"
    payload = {
        "gate": gate,
        "decision": decision,
        "comments": comments,
        "recorded_by": tool_context.agent_name,
        "at": stamp,
    }
    version = await tool_context.save_artifact(
        filename,
        types.Part(text=json.dumps(payload, indent=2)),
        custom_metadata={
            "kind": "checkpoint",
            "title": f"Gate: {gate}",
            "summary": f"{gate}: {decision}",
            "created_by": tool_context.agent_name,
        },
    )
    return {"filename": filename, "version": version}


async def update_board(items: list[dict], tool_context: ToolContext) -> dict[str, Any]:
    """Update the project kanban board (plan/board.json) — call at stage transitions.

    Args:
        items: Full board state, one dict per item:
            {"id": "lit-1", "title": "...",
             "type": "lit_task|hypothesis|experiment|analysis|report_section",
             "status": "backlog|in_progress|blocked|awaiting_review|done|killed",
             "status_reason": "...", "artifact_refs": ["lit/facet_1/notes.md"]}
    """
    import json

    version = await tool_context.save_artifact(
        "plan/board.json",
        types.Part(text=json.dumps({"items": items}, indent=2)),
        custom_metadata={
            "kind": "board",
            "title": "Project board",
            "summary": "; ".join(
                f"{i.get('id')}:{i.get('status')}" for i in items
            )[:200],
            "created_by": tool_context.agent_name,
        },
    )
    return {"filename": "plan/board.json", "version": version}


async def save_plan(
    plan_markdown: str, lit_facets: list[str], tool_context: ToolContext
) -> dict[str, Any]:
    """Save the research plan and register literature-search facets.

    Args:
        plan_markdown: Full plan document (objectives, key questions, lit facets,
            success criteria, report outline).
        lit_facets: 2-4 distinct literature-search facets, each a one-line
            description of a sub-question for one literature searcher.

    Returns:
        {filename, version, facet_count}.
    """
    from ..config import get_settings

    facets = [f.strip() for f in lit_facets if f.strip()][: get_settings().max_lit_facets]
    version = await tool_context.save_artifact(
        "plan/plan.md",
        types.Part(text=plan_markdown),
        custom_metadata={
            "kind": "plan",
            "title": "Research plan",
            "summary": f"Plan v{0} with {len(facets)} literature facets",
            "created_by": tool_context.agent_name,
        },
    )
    tool_context.state["lit_facets"] = facets
    for i in range(get_settings().max_lit_facets):
        tool_context.state[f"facet_{i + 1}"] = facets[i] if i < len(facets) else ""
    return {"filename": "plan/plan.md", "version": version, "facet_count": len(facets)}
