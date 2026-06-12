"""Logging + tracing setup (design §9: local-first, optional Langfuse).

Two independent pieces, both wired by ``setup_observability()`` (idempotent;
called from runner/gateway startup):

- **Logging** — root config with console + rotating file handler at
  ``DATA_ROOT/logs/deep_researcher.log`` (level via ``LOG_LEVEL``).
- **Tracing** — ADK already instruments every agent/LLM/tool call with
  OpenTelemetry spans, but they are no-ops until a tracer provider exists.
  When Langfuse keys are configured, spans export over OTLP/HTTP to
  ``<langfuse_host>/api/public/otel/v1/traces`` (Basic auth from
  public:secret keys). Without keys, tracing stays a no-op.
"""

from __future__ import annotations

import base64
import logging
import logging.handlers
from typing import Optional

from .config import Settings, get_settings

_configured = False


def langfuse_otlp_config(settings: Settings) -> Optional[dict[str, str]]:
    """(endpoint, headers) for Langfuse's OTLP trace ingestion, or None."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    token = base64.b64encode(
        f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
    ).decode()
    return {
        "endpoint": f"{settings.langfuse_host.rstrip('/')}/api/public/otel/v1/traces",
        "authorization": f"Basic {token}",
    }


def _setup_logging(settings: Settings) -> None:
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
    log_dir = settings.root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "deep_researcher.log", maxBytes=10_000_000, backupCount=5
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    # third-party chatter stays out of the way at default level
    for noisy in ("httpx", "LiteLLM", "litellm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def provider_from_model(model_id: object) -> Optional[str]:
    """LiteLLM model ids are 'provider/model' — e.g. 'deepseek/deepseek-chat'."""
    if isinstance(model_id, str) and "/" in model_id:
        return model_id.split("/", 1)[0]
    return None


def make_attribute_rewriter(default_system: str):
    """Span exporter wrapper that fixes ADK's Google-centric span attributes.

    ADK hardcodes ``gen_ai.system`` to 'gemini'/'vertex_ai' regardless of the
    actual model, and namespaces its attributes under ``gcp.vertex.agent.*``
    (google/adk telemetry has no override hook). This stack is local DeepSeek
    via LiteLLM, so before export we (a) set gen_ai.system from the span's own
    gen_ai.request.model, and (b) rename gcp.vertex.agent.* keys to adk.*.
    """
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope

    _WRONG_SYSTEMS = {"gemini", "vertex_ai", "gcp.vertex.agent"}
    _VERTEX_PREFIX = "gcp.vertex.agent."
    scope = InstrumentationScope(name="deep_researcher")

    class AttributeRewriter(SpanExporter):
        def __init__(self, delegate: SpanExporter):
            self._delegate = delegate

        def export(self, spans) -> SpanExportResult:
            for span in spans:
                attrs = dict(span.attributes or {})
                rewritten = {
                    (f"adk.{k[len(_VERTEX_PREFIX):]}" if k.startswith(_VERTEX_PREFIX) else k): v
                    for k, v in attrs.items()
                }
                if rewritten.get("gen_ai.system") in _WRONG_SYSTEMS:
                    rewritten["gen_ai.system"] = (
                        provider_from_model(rewritten.get("gen_ai.request.model"))
                        or default_system
                    )
                try:  # swap mislabeled scope ('gcp.vertex.agent') as well
                    if (span.instrumentation_scope or scope).name.startswith("gcp."):
                        span._instrumentation_scope = scope
                    span._attributes = rewritten
                except AttributeError:  # ReadableSpan internals moved; export as-is
                    pass
            return self._delegate.export(spans)

        def shutdown(self) -> None:
            self._delegate.shutdown()

        def force_flush(self, timeout_millis: int = 30_000) -> bool:
            return self._delegate.force_flush(timeout_millis)

    return AttributeRewriter


def _setup_tracing(settings: Settings) -> bool:
    config = langfuse_otlp_config(settings)
    if config is None:
        return False
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.app_name})
    )
    default_system = provider_from_model(settings.orchestrator_model) or "litellm"
    rewriter_cls = make_attribute_rewriter(default_system)
    provider.add_span_processor(
        BatchSpanProcessor(
            rewriter_cls(
                OTLPSpanExporter(
                    endpoint=config["endpoint"],
                    headers={
                        "Authorization": config["authorization"],
                        "x-langfuse-ingestion-version": "4",
                    },
                )
            )
        )
    )
    trace.set_tracer_provider(provider)
    return True


def setup_observability() -> None:
    """Configure logging and (when Langfuse keys exist) trace export. Idempotent."""
    global _configured
    if _configured:
        return
    _configured = True
    settings = get_settings()
    _setup_logging(settings)
    if _setup_tracing(settings):
        logging.getLogger(__name__).info(
            "tracing → Langfuse at %s", settings.langfuse_host
        )
