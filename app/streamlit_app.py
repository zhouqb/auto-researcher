"""Phase 0 steering UI (design §12.3): chat + gate buttons + artifact browser.

Run with:  uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import asyncio
import json

import streamlit as st

from deep_researcher.config import get_settings
from deep_researcher.runner import (
    build_runner,
    event_text,
    event_tool_calls,
    get_or_create_session,
    list_projects,
    run_turn,
    slugify,
)
from deep_researcher.monitor import list_runs, load_budget
from deep_researcher.storage import ArtifactCatalog

st.set_page_config(page_title="Deep Researcher", layout="wide")
settings = get_settings()


# -- async plumbing (fresh runner per call; engine binds to the call's loop) --

def fetch_projects() -> list[str]:
    async def _go():
        return [s.id for s in await list_projects(build_runner())]
    return asyncio.run(_go())


def fetch_history(project_id: str) -> list[tuple[str, str]]:
    async def _go():
        session = await get_or_create_session(build_runner(), project_id)
        return [
            (ev.author, event_text(ev))
            for ev in session.events
            if event_text(ev) and not ev.partial
        ]
    return asyncio.run(_go())


def send_message(project_id: str, message: str, status) -> None:
    async def _go():
        async for ev in run_turn(build_runner(), project_id, message):
            for tool in event_tool_calls(ev):
                status.write(f"`{ev.author}` → **{tool}**")
            text = event_text(ev)
            if text and not ev.partial:
                status.write(f"`{ev.author}`: {text[:300]}")
    asyncio.run(_go())


# -- sidebar: project selection + artifact browser ----------------------------

with st.sidebar:
    st.title("🔬 Deep Researcher")
    st.caption(f"data: `{settings.root}`")

    projects = fetch_projects()
    labels = ["➕ New project…"] + projects
    default_ix = (
        labels.index(st.session_state["project"])
        if st.session_state.get("project") in labels
        else 0
    )
    choice = st.selectbox("Project", labels, index=default_ix)

    if choice == "➕ New project…":
        question = st.text_area(
            "Research question",
            placeholder="e.g. How do MoE routing strategies affect inference throughput?",
        )
        if st.button("Start project", type="primary") and question.strip():
            st.session_state["project"] = slugify(question)
            st.session_state["pending_message"] = question.strip()
            st.rerun()
        project_id = None
    else:
        project_id = choice
        st.session_state["project"] = project_id

    if project_id:
        st.divider()
        st.subheader("Artifacts")
        catalog = ArtifactCatalog(settings.db_path)
        records = catalog.list_latest(project_id=project_id)
        if not records:
            st.caption("No artifacts yet.")
        for rec in records:
            label = f"{rec.title or rec.path}  ·  v{rec.version} ({rec.kind})"
            if st.button(label, key=f"art_{rec.id}", use_container_width=True):
                st.session_state["view_artifact"] = rec.path

# -- main area ----------------------------------------------------------------

if not st.session_state.get("project"):
    st.info("Create a project in the sidebar to begin.")
    st.stop()

project_id = st.session_state["project"]
chat_col, view_col = st.columns([3, 2], gap="large")

with chat_col:
    st.subheader(project_id)

    for author, text in fetch_history(project_id):
        role = "user" if author == "user" else "assistant"
        with st.chat_message(role):
            if role == "assistant":
                st.caption(author)
            st.markdown(text)

    # Gate quick actions (Gate 1: plan approval flows through chat).
    a, b, _sp = st.columns([1, 1, 2])
    quick = None
    if a.button("✅ Approve plan"):
        quick = "I approve the plan. Proceed."
    if b.button("✏️ Request changes"):
        st.session_state["show_revise"] = True
    if st.session_state.get("show_revise"):
        revision = st.text_input("What should change?", key="revise_text")
        if st.button("Send revision request") and revision.strip():
            quick = f"Please revise the plan: {revision.strip()}"
            st.session_state["show_revise"] = False

    typed = st.chat_input("Message the orchestrator…")
    message = st.session_state.pop("pending_message", None) or quick or typed

    if message:
        with st.chat_message("user"):
            st.markdown(message)
        with st.status("Working…", expanded=True) as status:
            try:
                send_message(project_id, message, status)
                status.update(label="Done", state="complete", expanded=False)
            except Exception as e:  # surface, don't bury, model/tool errors
                status.update(label=f"Error: {e}", state="error")
        st.rerun()

@st.fragment(run_every="3s")
def run_monitor(pid: str) -> None:
    """Live per-run view tailing codex_events.jsonl (design §12 run monitor)."""
    runs = list_runs(pid)
    budget = load_budget(pid)
    if budget and budget.get("totals"):
        t = budget["totals"]
        st.caption(
            f"**Budget** — runs: {len(budget['entries'])} · "
            f"in: {t.get('input_tokens', 0):,} tok · "
            f"out: {t.get('output_tokens', 0):,} tok · "
            f"wallclock: {t.get('wallclock_s', 0)}s"
        )
    if not runs:
        st.caption("No experiment runs yet.")
        return
    icons = {"running": "🟡", "completed": "🟢", "failed": "🔴", "timeout": "🟠"}
    for run in runs:
        head = (
            f"{icons.get(run.status, '⚪')} `{run.experiment}` run "
            f"`{run.run_id}` — **{run.status}**"
        )
        with st.expander(head, expanded=(run.status == "running")):
            if run.usage:
                st.caption(
                    f"tokens in/out: {run.usage.get('input_tokens', 0):,}/"
                    f"{run.usage.get('output_tokens', 0):,} · "
                    f"{run.wallclock_s}s · thread `{run.thread_id}`"
                )
            if run.metrics:
                st.json(run.metrics)
            for cmd in run.commands[-8:]:
                st.code(cmd, language="bash")
            if run.files_changed:
                st.caption("files: " + ", ".join(run.files_changed[-10:]))
            if run.last_message:
                st.markdown(run.last_message)


with view_col:
    monitor_tab, artifact_tab = st.tabs(["🖥 Runs", "📄 Artifact"])
    with monitor_tab:
        run_monitor(project_id)
    path = st.session_state.get("view_artifact")
    with artifact_tab:
        if path:
            st.subheader(path)
            file = settings.projects_dir / project_id / path
            if not file.exists():
                st.warning("File not found on disk.")
            elif path.endswith((".png", ".jpg", ".jpeg", ".gif")):
                st.image(str(file))
            elif path.endswith(".json"):
                st.json(json.loads(file.read_text()))
            else:
                st.markdown(file.read_text())
            st.download_button("Download", file.read_bytes(), file_name=file.name)
        else:
            st.caption("Select an artifact in the sidebar to view it here.")
