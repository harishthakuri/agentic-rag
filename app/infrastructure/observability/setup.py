"""OpenTelemetry setup for one process (API, worker or evals).

    app ──OTLP/HTTP──► Grafana LGTM  (OTLP_ENDPOINT):      traces, metrics, logs
        └─OTLP/HTTP──► Langfuse      (LANGFUSE_BASE_URL):  traces only

- Grafana gets every span (HTTP requests, SQL, ours) but never content:
  prompt/answer attributes are removed before export.
- Langfuse gets our spans and the HTTP request span above them, not the SQL
  spans: in an LLM trace view they would only be noise.
- Export runs in background threads, in batches. If a destination is down, the
  exporter logs a warning and drops the data; requests are never affected.

Everything here is process-global (OpenTelemetry has one tracer and meter provider
per process), so `configure` is called once at startup, and the returned function
flushes and shuts down at exit.
"""

import base64
import logging
from collections.abc import Callable, Sequence
from typing import Any

import requests
from opentelemetry import context, metrics, trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.sampling import (
    Decision,
    ParentBased,
    Sampler,
    SamplingResult,
    TraceIdRatioBased,
)
from opentelemetry.trace import Link, SpanKind
from opentelemetry.util.types import Attributes

from app.core.config import Settings
from app.core.logging import json_formatter
from app.infrastructure.observability.adapter import CONTENT_ATTRIBUTES, SCOPE

logger = logging.getLogger(__name__)

APP_VERSION = "0.1.0"
METRIC_EXPORT_INTERVAL_MS = 10_000
# Spans Langfuse receives: ours, and the HTTP request span that is their parent.
LANGFUSE_SCOPES = frozenset({SCOPE, "opentelemetry.instrumentation.fastapi"})


def configure(settings: Settings, service_name: str) -> Callable[[], None]:
    """Set the global providers and exporters. Returns a function that flushes and
    shuts them down."""
    if not settings.observability_enabled:
        return lambda: None
    if not settings.otlp_endpoint and not settings.langfuse_base_url:
        logger.warning(
            "OBSERVABILITY_ENABLED is set, but neither OTLP_ENDPOINT nor "
            "LANGFUSE_BASE_URL is: telemetry is recorded but sent nowhere"
        )

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": APP_VERSION,
            "deployment.environment.name": settings.observability_environment,
        }
    )
    tracer_provider = build_tracer_provider(settings, resource)
    meter_provider = build_meter_provider(settings, resource)
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)

    shutdowns: list[Callable[[], Any]] = [tracer_provider.shutdown, meter_provider.shutdown]
    if settings.otlp_endpoint:
        shutdowns.append(_export_logs(settings.otlp_endpoint, resource))

    logger.info(
        "Observability on: grafana=%s langfuse=%s content=%s",
        settings.otlp_endpoint or "-",
        settings.langfuse_base_url or "-",
        settings.observability_capture_content,
    )

    def shutdown() -> None:
        for close in shutdowns:
            try:
                close()
            except Exception:  # never fail the shutdown of the app over telemetry
                logger.warning("Telemetry shutdown failed", exc_info=True)

    return shutdown


def build_tracer_provider(settings: Settings, resource: Resource) -> TracerProvider:
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(root=SkipOrphanClientCalls(settings.observability_sample_ratio)),
    )
    if settings.otlp_endpoint:
        grafana = OTLPSpanExporter(
            endpoint=_url(settings.otlp_endpoint, "/v1/traces"), session=fresh_connections()
        )
        provider.add_span_processor(BatchSpanProcessor(WithoutContent(grafana)))
    if settings.langfuse_base_url:
        langfuse = OTLPSpanExporter(
            endpoint=_url(settings.langfuse_base_url, "/api/public/otel/v1/traces"),
            headers=langfuse_headers(
                settings.langfuse_public_key, settings.langfuse_secret_key.get_secret_value()
            ),
            session=fresh_connections(),
        )
        provider.add_span_processor(BatchSpanProcessor(OnlyScopes(langfuse, LANGFUSE_SCOPES)))
    return provider


def build_meter_provider(settings: Settings, resource: Resource) -> MeterProvider:
    readers = []
    if settings.otlp_endpoint:  # Langfuse doesn't ingest metrics
        exporter = OTLPMetricExporter(
            endpoint=_url(settings.otlp_endpoint, "/v1/metrics"), session=fresh_connections()
        )
        readers.append(
            PeriodicExportingMetricReader(
                exporter, export_interval_millis=METRIC_EXPORT_INTERVAL_MS
            )
        )
    return MeterProvider(resource=resource, metric_readers=readers)


def fresh_connections() -> requests.Session:
    """An HTTP session that opens a new connection for every export batch.

    Exporters send a batch every few seconds, so by default they keep connections
    alive and reuse them. Through Docker's port forwarding (seen with Rancher
    Desktop) such idle connections can die silently: the app still sees them as
    open, an export waits on one forever, and that process stops sending telemetry
    while new processes work fine. A fresh connection per batch costs well under a
    millisecond locally and cannot go stale.
    """
    session = requests.Session()
    session.headers["Connection"] = "close"
    return session


def langfuse_headers(public_key: str, secret_key: str) -> dict[str, str]:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return {"Authorization": f"Basic {token}", "x-langfuse-ingestion-version": "4"}


def instrument_app(app: Any) -> None:
    """Server spans and HTTP metrics for FastAPI. Health checks and the static UI
    are skipped, and so are the per-message ASGI send/receive spans (a streamed
    answer would otherwise produce one span per token)."""
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(
        app, excluded_urls="/health/,/ui/", exclude_spans=["receive", "send"]
    )


def instrument_database(engine: Any, *, tracer_provider: TracerProvider | None = None) -> None:
    """A span per SQL statement (Grafana only)."""
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    # The instrumentation declares support up to SQLAlchemy 2.0 and refuses 2.1 without
    # this flag; the engine events it relies on are unchanged in 2.1
    # (tests/integration/test_observability.py checks it).
    SQLAlchemyInstrumentor().instrument(
        engine=engine.sync_engine, tracer_provider=tracer_provider, skip_dep_check=True
    )


def _export_logs(endpoint: str, resource: Resource) -> Callable[[], None]:
    """Also send log records to Grafana (Loki). Each carries the trace and span ID
    of the request it belongs to; the body is the same JSON as the console line."""
    provider = LoggerProvider(resource=resource)
    provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(endpoint=_url(endpoint, "/v1/logs"), session=fresh_connections())
        )
    )
    handler = _LoggingHandler(level=logging.INFO, logger_provider=provider)
    handler.setFormatter(json_formatter())
    logging.getLogger().addHandler(handler)
    return provider.shutdown


class _LoggingHandler(LoggingHandler):
    """Leaves out private record fields: structlog attaches its logger object as
    `_logger`, which is not a valid attribute value."""

    @staticmethod
    def _get_attributes(record: logging.LogRecord) -> dict[str, Any]:
        attributes = LoggingHandler._get_attributes(record)
        return {k: v for k, v in (attributes or {}).items() if not k.startswith("_")}


def _url(base: str, path: str) -> str:
    return base.rstrip("/") + path


# --- Sampling -------------------------------------------------------------------
class SkipOrphanClientCalls(Sampler):
    """Decides for spans that start a new trace.

    Outgoing calls (SQL statements, HTTP requests) that are not part of a request or
    job are not traced: the worker polls the queue every few seconds and health
    checks hit the database, and each would otherwise become a one-span trace.
    Everything else is sampled at `ratio`.
    """

    def __init__(self, ratio: float) -> None:
        self._ratio = TraceIdRatioBased(ratio)

    def should_sample(
        self,
        parent_context: context.Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        trace_state: trace.TraceState | None = None,
    ) -> SamplingResult:
        if kind is SpanKind.CLIENT:
            return SamplingResult(Decision.DROP)
        return self._ratio.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

    def get_description(self) -> str:
        return f"SkipOrphanClientCalls({self._ratio.get_description()})"


# --- Exporter filters -----------------------------------------------------------
class WithoutContent(SpanExporter):
    """Removes prompt/answer attributes, so they never reach this destination."""

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._inner.export([_without_content(s) for s in spans])

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)


class OnlyScopes(SpanExporter):
    """Exports only spans created by the given instrumentation scopes."""

    def __init__(self, inner: SpanExporter, scopes: frozenset[str]) -> None:
        self._inner = inner
        self._scopes = scopes

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        kept = [
            s
            for s in spans
            if s.instrumentation_scope and s.instrumentation_scope.name in self._scopes
        ]
        return self._inner.export(kept) if kept else SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)


def _without_content(span: ReadableSpan) -> ReadableSpan:
    attributes = span.attributes or {}
    if not CONTENT_ATTRIBUTES & attributes.keys():
        return span
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes={k: v for k, v in attributes.items() if k not in CONTENT_ATTRIBUTES},
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )
