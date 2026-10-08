"""The telemetry port implemented with OpenTelemetry.

Spans carry two families of attributes on top of the caller's:

- Langfuse's: `langfuse.observation.type` (from the span kind), `langfuse.trace.name`
  on the span that names a trace, and, with content capture on,
  `langfuse.observation.input` / `.output` (JSON, long strings truncated).
- The caller's own, e.g. the OpenTelemetry GenAI conventions (`gen_ai.*`) set by the
  model adapters, and ours (`rag.*`).

Content attributes are stripped before spans go to Grafana (see setup.py), so only
Langfuse ever stores document text.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.metrics import Counter, Histogram, MeterProvider
from opentelemetry.trace import Status, StatusCode, TracerProvider

from app.application.ports.telemetry import (
    AGENT_TOOL_CALLS,
    ANSWERS,
    INGESTION_DURATION,
    INGESTION_JOBS,
    LLM_DURATION,
    LLM_TOKENS,
    PARSE_FALLBACKS,
    RERANK_FAILURES,
    STEP_DURATION,
    Attributes,
    Span,
    SpanKind,
)

SCOPE = "agentic-rag"  # the instrumentation scope of our own spans and metrics

CONTENT_ATTRIBUTES = frozenset({"langfuse.observation.input", "langfuse.observation.output"})

_SECONDS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120, 300)
_TOKENS = (16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
_HISTOGRAMS: dict[str, tuple[str, Sequence[float]]] = {
    STEP_DURATION: ("s", _SECONDS),
    LLM_DURATION: ("s", _SECONDS),
    LLM_TOKENS: ("{token}", _TOKENS),
    AGENT_TOOL_CALLS: ("{call}", (0, 1, 2, 3, 4, 5, 6, 8, 10, 12)),
    INGESTION_DURATION: ("s", _SECONDS),
}
_COUNTERS: dict[str, str] = {
    ANSWERS: "{answer}",
    RERANK_FAILURES: "{failure}",
    INGESTION_JOBS: "{job}",
    PARSE_FALLBACKS: "{document}",
}


class OpenTelemetryAdapter:
    def __init__(
        self,
        *,
        capture_content: bool = False,
        max_content_chars: int = 8000,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
    ) -> None:
        self._capture = capture_content
        self._max_chars = max_content_chars
        # None: the global providers (set by setup.configure); tests pass their own.
        self._tracer = trace.get_tracer(SCOPE, tracer_provider=tracer_provider)
        self._meter = metrics.get_meter(SCOPE, meter_provider=meter_provider)
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}

    @property
    def capture_content(self) -> bool:
        return self._capture

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: SpanKind = "span",
        trace_name: str | None = None,
        attributes: Attributes | None = None,
    ) -> Iterator[Span]:
        with self._tracer.start_as_current_span(
            name, attributes=_attributes(kind, trace_name, attributes)
        ) as span:
            yield _OtelSpan(span, self)

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = "span",
        trace_name: str | None = None,
        attributes: Attributes | None = None,
    ) -> Span:
        span = self._tracer.start_span(name, attributes=_attributes(kind, trace_name, attributes))
        return _OtelSpan(span, self)

    def count(self, name: str, value: int = 1, labels: Mapping[str, str] | None = None) -> None:
        counter = self._counters.get(name)
        if counter is None:
            counter = self._meter.create_counter(name, unit=_COUNTERS.get(name, ""))
            self._counters[name] = counter
        counter.add(value, dict(labels or {}))

    def record(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        histogram = self._histograms.get(name)
        if histogram is None:
            unit, buckets = _HISTOGRAMS.get(name, ("", _SECONDS))
            histogram = self._meter.create_histogram(
                name, unit=unit, explicit_bucket_boundaries_advisory=buckets
            )
            self._histograms[name] = histogram
        histogram.record(value, dict(labels or {}))

    def _content(self, value: Any) -> str:
        return json.dumps(_truncate(value, self._max_chars), ensure_ascii=False, default=str)


class _OtelSpan:
    def __init__(self, span: trace.Span, owner: OpenTelemetryAdapter) -> None:
        self._span = span
        self._owner = owner

    def set(self, attributes: Attributes) -> None:
        self._span.set_attributes({k: v for k, v in attributes.items() if v is not None})

    def content(self, *, input: Any = None, output: Any = None) -> None:
        if not self._owner.capture_content:
            return
        if input is not None:
            self._span.set_attribute("langfuse.observation.input", self._owner._content(input))
        if output is not None:
            self._span.set_attribute("langfuse.observation.output", self._owner._content(output))

    def activate(self) -> AbstractContextManager[None]:
        return _activate(self._span)

    def fail(self, error: BaseException) -> None:
        self._span.record_exception(error)
        self._span.set_status(Status(StatusCode.ERROR, f"{type(error).__name__}: {error}"))

    def end(self) -> None:
        self._span.end()


@contextmanager
def _activate(span: trace.Span) -> Iterator[None]:
    with trace.use_span(span, end_on_exit=False):
        yield


def _attributes(
    kind: SpanKind, trace_name: str | None, attributes: Attributes | None
) -> dict[str, Any]:
    result: dict[str, Any] = {k: v for k, v in (attributes or {}).items() if v is not None}
    if kind != "span":
        result["langfuse.observation.type"] = kind
    if trace_name:
        result["langfuse.trace.name"] = trace_name
    return result


def _truncate(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"… [{len(value) - limit} more]"
    if isinstance(value, Mapping):
        return {str(k): _truncate(v, limit) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_truncate(v, limit) for v in value]
    return value
