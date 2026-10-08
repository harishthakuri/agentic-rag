import asyncio
import base64
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from uuid import UUID

import httpx2 as httpx  # the OpenAI SDK is built on httpx2
import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import Decision, ParentBased
from opentelemetry.trace import SpanKind

from app.application.ports.chat import ChatMessage, ToolCall
from app.application.ports.telemetry import (
    AGENT_TOOL_CALLS,
    ANSWERS,
    INGESTION_JOBS,
    LLM_TOKENS,
    STEP_DURATION,
)
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.use_cases.agent import AgentAsk, AgentQuery
from app.application.use_cases.answering import AskQuery, AskQuestion, ContextBuilder
from app.application.use_cases.collections import CreateCollection, CreateCollectionCommand
from app.application.use_cases.ingestion import ProcessNextIngestionJob, UploadDocumentCommand
from app.application.use_cases.retrieval import SearchCollection
from app.core.config import Settings
from app.infrastructure.chunking.structure_aware import StructureAwareChunker
from app.infrastructure.observability import OpenTelemetryAdapter, configure
from app.infrastructure.observability.adapter import SCOPE
from app.infrastructure.observability.setup import (
    OnlyScopes,
    SkipOrphanClientCalls,
    WithoutContent,
    build_tracer_provider,
    fresh_connections,
    langfuse_headers,
)
from app.infrastructure.parsing import DefaultParserRegistry
from tests.fakes import (
    FakeChatModel,
    FakeEmbedder,
    InMemoryFileStorage,
    InMemoryStore,
    InMemoryUnitOfWork,
)
from tests.unit.test_adapters import _chat_model, _sse_chunks
from tests.unit.test_chunker import WordCounter
from tests.unit.test_ingestion import GUIDE, _upload
from tests.unit.test_retrieval import SPEC, seed_collection


@dataclass
class Recorder:
    """An adapter whose spans and metrics stay in memory."""

    telemetry: OpenTelemetryAdapter
    exporter: InMemorySpanExporter
    reader: InMemoryMetricReader
    provider: TracerProvider

    def spans(self) -> dict[str, ReadableSpan]:
        return {s.name: s for s in self.exporter.get_finished_spans()}

    def points(self, metric: str) -> list[Any]:
        data = self.reader.get_metrics_data()
        return [
            point
            for resource in (data.resource_metrics if data else [])
            for scope in resource.scope_metrics
            for m in scope.metrics
            if m.name == metric
            for point in m.data.data_points
        ]


def _recorder(*, capture: bool = False) -> Recorder:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    telemetry = OpenTelemetryAdapter(
        capture_content=capture,
        tracer_provider=provider,
        meter_provider=MeterProvider(metric_readers=[reader]),
    )
    return Recorder(telemetry, exporter, reader, provider)


def _parent(span: ReadableSpan) -> int | None:
    return span.parent.span_id if span.parent else None


def _id(span: ReadableSpan) -> int:
    assert span.context is not None
    span_id: int = span.context.span_id
    return span_id


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def uow(store: InMemoryStore) -> UnitOfWorkFactory:
    return lambda: InMemoryUnitOfWork(store)


@pytest.fixture
async def collection_id(store: InMemoryStore) -> UUID:
    return (await seed_collection(store, FakeEmbedder(SPEC))).id


def _ask(uow: UnitOfWorkFactory, chat: FakeChatModel, recorder: Recorder) -> AskQuestion:
    search = SearchCollection(uow, FakeEmbedder(SPEC), telemetry=recorder.telemetry)
    return AskQuestion(search, chat, ContextBuilder(WordCounter()), telemetry=recorder.telemetry)


# --- Use cases ------------------------------------------------------------------
async def test_ask_produces_a_trace_tree_and_metrics(
    uow: UnitOfWorkFactory, collection_id: UUID
) -> None:
    recorder = _recorder()
    await _ask(uow, FakeChatModel("Ingress routes HTTP [1]."), recorder).execute(
        AskQuery(collection_id, "ingress routes http")
    )

    spans = recorder.spans()
    root = spans["rag.ask"]
    assert root.parent is None
    for child in ("rag.search", "rag.context", "rag.grounding"):
        assert _parent(spans[child]) == _id(root)
    for step in ("rag.vector_search", "rag.keyword_search"):
        assert _parent(spans[step]) == _id(spans["rag.search"])

    assert root.attributes is not None
    assert root.attributes["langfuse.trace.name"] == "ask"
    assert root.attributes["langfuse.observation.type"] == "chain"
    assert root.attributes["rag.answer.outcome"] == "cited"
    assert (spans["rag.search"].attributes or {})["langfuse.observation.type"] == "retriever"
    assert (spans["rag.grounding"].attributes or {})["langfuse.observation.type"] == "guardrail"
    assert "langfuse.observation.input" not in root.attributes  # content capture is off

    [answers] = recorder.points(ANSWERS)
    assert dict(answers.attributes) == {"mode": "ask", "outcome": "cited"}
    assert answers.value == 1
    steps = {p.attributes["step"] for p in recorder.points(STEP_DURATION)}
    assert {"embed", "vector_search", "keyword_search", "generate"} <= steps


async def test_content_is_recorded_only_when_capture_is_on(
    uow: UnitOfWorkFactory, collection_id: UUID
) -> None:
    recorder = _recorder(capture=True)
    await _ask(uow, FakeChatModel("Ingress routes HTTP [1]."), recorder).execute(
        AskQuery(collection_id, "ingress routes http")
    )

    attributes = recorder.spans()["rag.ask"].attributes or {}
    assert json.loads(str(attributes["langfuse.observation.input"])) == "ingress routes http"
    assert json.loads(str(attributes["langfuse.observation.output"])) == "Ingress routes HTTP [1]."


async def test_a_streamed_answer_can_resume_in_another_task(
    uow: UnitOfWorkFactory, collection_id: UUID, caplog: pytest.LogCaptureFixture
) -> None:
    # Streaming responses read the first event in the request task and the rest in
    # another one. No span may stay "current" across that switch.
    recorder = _recorder()
    events = aiter(
        _ask(uow, FakeChatModel("Ingress routes HTTP [1]."), recorder).stream(
            AskQuery(collection_id, "ingress routes http")
        )
    )
    await anext(events)

    async def rest() -> list[object]:
        return [e async for e in events]

    with caplog.at_level(logging.ERROR, logger="opentelemetry.context"):
        await asyncio.create_task(rest())

    assert not caplog.records  # e.g. "Failed to detach context"
    spans = recorder.spans()
    assert _parent(spans["rag.grounding"]) == _id(spans["rag.ask"])


def _call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=json.dumps(arguments))


async def test_agent_tool_calls_contain_their_search(
    uow: UnitOfWorkFactory, collection_id: UUID
) -> None:
    recorder = _recorder()
    chat = FakeChatModel(
        [[_call("search_knowledge_base", query="ingress routes http")], "Ingress routes it [1]."]
    )
    search = SearchCollection(uow, FakeEmbedder(SPEC), telemetry=recorder.telemetry)
    agent = AgentAsk(uow, search, chat, telemetry=recorder.telemetry)
    await agent.execute(AgentQuery(collection_id, "ingress?"))

    spans = recorder.spans()
    root = spans["rag.agent"]
    tool = spans["execute_tool search_knowledge_base"]
    assert root.parent is None and _parent(tool) == _id(root)
    assert _parent(spans["rag.search"]) == _id(tool)
    assert (tool.attributes or {})["langfuse.observation.type"] == "tool"
    [answers] = recorder.points(ANSWERS)
    assert dict(answers.attributes) == {"mode": "agent", "outcome": "cited"}
    [tool_calls] = recorder.points(AGENT_TOOL_CALLS)
    assert tool_calls.sum == 1


async def test_ingestion_job_steps_are_traced(uow: UnitOfWorkFactory) -> None:
    recorder = _recorder()
    storage = InMemoryFileStorage()
    collection = await CreateCollection(uow, SPEC).execute(CreateCollectionCommand(name="docs"))
    await _upload(uow, storage).execute(UploadDocumentCommand(collection.id, "guide.md", GUIDE))
    processor = ProcessNextIngestionJob(
        uow,
        storage,
        DefaultParserRegistry(),
        StructureAwareChunker(WordCounter(), target_tokens=100, overlap_tokens=10),
        FakeEmbedder(SPEC),
        lease=timedelta(minutes=15),
        telemetry=recorder.telemetry,
    )

    assert (outcome := await processor.execute("worker-1")) is not None and outcome.succeeded

    spans = recorder.spans()
    job = spans["ingestion.job"]
    assert (job.attributes or {})["langfuse.trace.name"] == "ingestion"
    for step in ("ingestion.parse", "ingestion.chunk", "ingestion.embed", "ingestion.store"):
        assert _parent(spans[step]) == _id(job)
    assert (spans["ingestion.parse"].attributes or {})["rag.parser"] == "markdown"
    [jobs] = recorder.points(INGESTION_JOBS)
    assert dict(jobs.attributes) == {
        "document_type": "markdown",
        "parser": "markdown",
        "status": "succeeded",
    }


async def test_chat_calls_are_generation_spans_with_token_usage() -> None:
    recorder = _recorder(capture=True)

    def handle(request: httpx.Request) -> httpx.Response:
        body = _sse_chunks(
            {"content": "A 304 means Not Modified [1]."},
            usage={"prompt_tokens": 120, "completion_tokens": 9},
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    model = _chat_model(httpx.MockTransport(handle), telemetry=recorder.telemetry)
    [_ async for _ in model.stream([ChatMessage("user", "What is 304?")])]

    attributes = recorder.spans()["chat gpt-oss:20b"].attributes or {}
    assert attributes["langfuse.observation.type"] == "generation"
    assert attributes["gen_ai.operation.name"] == "chat"
    assert (attributes["gen_ai.usage.input_tokens"], attributes["gen_ai.usage.output_tokens"]) == (
        120,
        9,
    )
    assert "What is 304?" in str(attributes["langfuse.observation.input"])
    tokens = {p.attributes["gen_ai.token.type"]: p.sum for p in recorder.points(LLM_TOKENS)}
    assert tokens == {"input": 120, "output": 9}


# --- Exporters and sampling -----------------------------------------------------
def _finished(recorder: Recorder) -> Sequence[ReadableSpan]:
    return recorder.exporter.get_finished_spans()


def test_grafana_never_receives_content() -> None:
    recorder = _recorder(capture=True)
    with recorder.telemetry.span("rag.ask", attributes={"rag.sources": 2}) as span:
        span.content(input="the question", output="the answer")
    grafana = InMemorySpanExporter()

    WithoutContent(grafana).export(_finished(recorder))

    [exported] = grafana.get_finished_spans()
    assert dict(exported.attributes or {}) == {"rag.sources": 2}


def test_langfuse_receives_only_selected_scopes() -> None:
    recorder = _recorder()
    with recorder.telemetry.span("rag.ask"):
        pass
    with recorder.provider.get_tracer(
        "opentelemetry.instrumentation.sqlalchemy"
    ).start_as_current_span("SELECT"):
        pass
    langfuse = InMemorySpanExporter()

    OnlyScopes(langfuse, frozenset({SCOPE})).export(_finished(recorder))

    assert [s.name for s in langfuse.get_finished_spans()] == ["rag.ask"]


def test_outgoing_calls_outside_a_request_are_not_traced() -> None:
    sampler = SkipOrphanClientCalls(1.0)
    assert sampler.should_sample(None, 1, "SELECT", SpanKind.CLIENT).decision is Decision.DROP
    assert sampler.should_sample(None, 1, "GET /x", SpanKind.SERVER).decision is (
        Decision.RECORD_AND_SAMPLE
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=ParentBased(root=SkipOrphanClientCalls(1.0)))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("SELECT polling", kind=SpanKind.CLIENT):
        pass
    with (
        tracer.start_as_current_span("ingestion.job"),
        tracer.start_as_current_span("SELECT chunks", kind=SpanKind.CLIENT),
    ):
        pass

    assert {s.name for s in exporter.get_finished_spans()} == {"ingestion.job", "SELECT chunks"}


def test_langfuse_authenticates_with_basic_auth() -> None:
    headers = langfuse_headers("pk-lf-1", "sk-lf-2")
    assert headers["Authorization"] == "Basic " + base64.b64encode(b"pk-lf-1:sk-lf-2").decode()


def test_exporters_open_a_fresh_connection_per_batch(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reused idle connections can die silently behind Docker's port forwarding and
    # then block an exporter for good (see setup.fresh_connections).
    from app.infrastructure.observability import setup

    sessions: list[Any] = []

    class RecordingExporter(InMemorySpanExporter):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            sessions.append(kwargs["session"])

    monkeypatch.setattr(setup, "OTLPSpanExporter", RecordingExporter)
    enabled = settings.model_copy(
        update={"otlp_endpoint": "http://localhost:4318", "langfuse_base_url": "http://lf"}
    )
    build_tracer_provider(enabled, Resource.create()).shutdown()

    assert len(sessions) == 2  # Grafana and Langfuse
    assert all(s.headers["Connection"] == "close" for s in sessions)
    assert fresh_connections().headers["Connection"] == "close"


def test_disabled_observability_sets_nothing_up(settings: Settings) -> None:
    shutdown = configure(settings.model_copy(update={"observability_enabled": False}), "test")
    shutdown()  # a no-op, and safe to call
