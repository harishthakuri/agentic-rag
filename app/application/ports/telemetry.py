"""Telemetry port: traces and metrics, without tying the use cases to a vendor.

The use cases describe what happens ("rag.search took 0.2 s and returned 8 hits");
the adapter (infrastructure/observability) turns that into OpenTelemetry spans and
metrics, which go to Grafana and/or Langfuse. With observability off, the no-op
implementation below is used and nothing is recorded.

Two ways to open a span:

- `with telemetry.span(...)` for code that only awaits. The span is the "current"
  one inside the block, so spans opened below it become its children.
- `telemetry.start_span(...)` in async generators (streamed answers), which yield
  to their caller and may resume in another task. The span must then not stay
  current across a `yield`: use `with span.activate():` around the awaits that
  should be its children, and call `span.end()` (in a `finally`).

`kind` says what a span is for (it maps to Langfuse's observation types):
a model call is a "generation", search and reranking are "retriever", and so on.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Literal, Protocol

AttributeValue = str | bool | int | float | Sequence[str]
Attributes = Mapping[str, AttributeValue | None]  # None values are skipped
SpanKind = Literal[
    "span", "chain", "generation", "embedding", "retriever", "tool", "agent", "guardrail"
]

# Metric names (histograms record seconds, tokens or counts; counters count events).
STEP_DURATION = "rag.step.duration"
ANSWERS = "rag.answers"
RERANK_FAILURES = "rag.rerank.failures"
AGENT_TOOL_CALLS = "rag.agent.tool_calls"
LLM_DURATION = "gen_ai.client.operation.duration"
LLM_TOKENS = "gen_ai.client.token.usage"
INGESTION_JOBS = "ingestion.jobs"
INGESTION_DURATION = "ingestion.job.duration"
PARSE_FALLBACKS = "ingestion.parse.fallbacks"


class Span(Protocol):
    def set(self, attributes: Attributes) -> None: ...

    def content(self, *, input: Any = None, output: Any = None) -> None:
        """Prompts, questions, passages or answers. Kept only when content capture is
        on, so callers can pass them freely."""
        ...

    def activate(self) -> AbstractContextManager[None]:
        """Make this span the current one for a block that doesn't `yield`."""
        ...

    def fail(self, error: BaseException) -> None: ...

    def end(self) -> None: ...


class Telemetry(Protocol):
    @property
    def capture_content(self) -> bool: ...

    def span(
        self,
        name: str,
        *,
        kind: SpanKind = "span",
        trace_name: str | None = None,
        attributes: Attributes | None = None,
    ) -> AbstractContextManager[Span]:
        """A span around a block; records an exception that escapes it."""
        ...

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = "span",
        trace_name: str | None = None,
        attributes: Attributes | None = None,
    ) -> Span:
        """A span the caller ends: for async generators (see the module docstring)."""
        ...

    def count(self, name: str, value: int = 1, labels: Mapping[str, str] | None = None) -> None: ...

    def record(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None: ...


# --- No-op: observability off, and the default in tests ------------------------
class NoopSpan:
    def set(self, attributes: Attributes) -> None:
        pass

    def content(self, *, input: Any = None, output: Any = None) -> None:
        pass

    @contextmanager
    def activate(self) -> Iterator[None]:
        yield

    def fail(self, error: BaseException) -> None:
        pass

    def end(self) -> None:
        pass


NOOP_SPAN = NoopSpan()


class NoopTelemetry:
    @property
    def capture_content(self) -> bool:
        return False

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: SpanKind = "span",
        trace_name: str | None = None,
        attributes: Attributes | None = None,
    ) -> Iterator[Span]:
        yield NOOP_SPAN

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = "span",
        trace_name: str | None = None,
        attributes: Attributes | None = None,
    ) -> Span:
        return NOOP_SPAN

    def count(self, name: str, value: int = 1, labels: Mapping[str, str] | None = None) -> None:
        pass

    def record(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        pass


NOOP_TELEMETRY = NoopTelemetry()
