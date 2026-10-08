"""SQL spans with the installed SQLAlchemy. The OpenTelemetry instrumentation only
declares support up to SQLAlchemy 2.0, so this checks it really works with ours."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.infrastructure.observability import instrument_database
from tests.integration.conftest import PostgresUrls

pytestmark = pytest.mark.integration


async def test_sql_statements_become_child_spans(migrated_postgres: PostgresUrls) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    engine = create_async_engine(migrated_postgres.app_sqlalchemy)
    instrument_database(engine, tracer_provider=provider)
    try:
        with provider.get_tracer("test").start_as_current_span("ingestion.job"):
            async with engine.connect() as connection:
                await connection.execute(text("SELECT count(*) FROM chunks"))
    finally:
        await engine.dispose()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    job = spans["ingestion.job"]
    queries = [s for s in spans.values() if s.name.lower().startswith("select")]
    assert queries, f"no SQL span among {list(spans)}"
    assert job.context is not None and queries[0].parent is not None
    assert queries[0].parent.span_id == job.context.span_id
