"""Observability tests: Langfuse OTLP config, logging setup, span flow."""

from __future__ import annotations

import base64
import logging

import pytest
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from scripted_llm import SCRIPTS, ScriptedLlm, _text, patch_models

import deep_researcher.config as config_mod
import deep_researcher.observability as obs
from deep_researcher.agents import build_root_agent
from deep_researcher.storage import ArtifactCatalog, LocalArtifactService

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-dummy")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    config_mod.get_settings.cache_clear()
    obs._configured = False
    yield
    obs._configured = False
    config_mod.get_settings.cache_clear()
    # drop the file handler added during the test so tmp dirs can be reaped
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.handlers.RotatingFileHandler):
            root.removeHandler(h)
            h.close()


def test_otlp_config_requires_both_keys(monkeypatch):
    settings = config_mod.get_settings()
    assert obs.langfuse_otlp_config(settings) is None

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-1")
    config_mod.get_settings.cache_clear()
    assert obs.langfuse_otlp_config(config_mod.get_settings()) is None  # secret missing

    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-2")
    monkeypatch.setenv("LANGFUSE_HOST", "http://lf.local:3000/")
    config_mod.get_settings.cache_clear()
    config = obs.langfuse_otlp_config(config_mod.get_settings())
    assert config["endpoint"] == "http://lf.local:3000/api/public/otel/v1/traces"
    expected = base64.b64encode(b"pk-lf-1:sk-lf-2").decode()
    assert config["authorization"] == f"Basic {expected}"


def test_setup_logging_writes_file_and_is_idempotent():
    settings = config_mod.get_settings()
    obs.setup_observability()
    obs.setup_observability()  # second call is a no-op
    logging.getLogger("deep_researcher.test").info("hello log")
    for h in logging.getLogger().handlers:
        h.flush()
    log_file = settings.root / "logs" / "deep_researcher.log"
    assert log_file.exists()
    assert "hello log" in log_file.read_text()
    # tracing not configured without keys
    assert obs._setup_tracing(settings) is False


async def test_adk_spans_flow_through_configured_provider(tmp_path):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)  # what _setup_tracing does, minus OTLP

    settings = config_mod.get_settings()
    SCRIPTS.clear()
    SCRIPTS.update({"orchestrator": [[_text("Q1: scope?")]]})
    root = build_root_agent()
    patch_models(root, ScriptedLlm(model="scripted"))
    runner = Runner(
        app=App(name=settings.app_name, root_agent=root),
        session_service=InMemorySessionService(),
        artifact_service=LocalArtifactService(
            settings.root, ArtifactCatalog(settings.db_path)
        ),
    )
    await runner.session_service.create_session(
        app_name=settings.app_name, user_id="local", session_id="p-otel"
    )
    async for _ in runner.run_async(
        user_id="local", session_id="p-otel",
        new_message=types.Content(role="user", parts=[types.Part(text="hi")]),
    ):
        pass

    names = [s.name for s in exporter.get_finished_spans()]
    assert names, "ADK should emit spans once a tracer provider is configured"
    assert any("orchestrator" in n or "call_llm" in n or "invocation" in n
               for n in names), names
