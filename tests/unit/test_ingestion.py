"""Upload and the ingestion pipeline, end to end over in-memory adapters."""

from datetime import UTC, datetime, timedelta

import pytest

from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.use_cases.collections import (
    CollectionNotFoundError,
    CreateCollection,
    CreateCollectionCommand,
)
from app.application.use_cases.documents import DuplicateDocumentError
from app.application.use_cases.ingestion import (
    ProcessNextIngestionJob,
    UploadDocument,
    UploadDocumentCommand,
)
from app.domain.exceptions import DomainValidationError
from app.domain.models import Collection, DocumentStatus, IngestionJob, JobStatus
from app.domain.value_objects import EmbeddingSpec, new_id
from app.infrastructure.chunking import StructureAwareChunker
from app.infrastructure.parsing import DefaultParserRegistry
from tests.fakes import FakeEmbedder, InMemoryFileStorage, InMemoryStore, InMemoryUnitOfWork
from tests.unit.test_chunker import WordCounter

SPEC = EmbeddingSpec(model="test-embedder", dimensions=8)

GUIDE = b"""# Kubernetes Guide

## Services

A Service gives pods a stable virtual IP.

## Ingress

Ingress routes external HTTP traffic.
"""


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def uow(store: InMemoryStore) -> UnitOfWorkFactory:
    return lambda: InMemoryUnitOfWork(store)


@pytest.fixture
def storage() -> InMemoryFileStorage:
    return InMemoryFileStorage()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(SPEC)


@pytest.fixture
async def collection(uow: UnitOfWorkFactory) -> Collection:
    return await CreateCollection(uow, SPEC).execute(CreateCollectionCommand(name="docs"))


def _upload(uow: UnitOfWorkFactory, storage: InMemoryFileStorage) -> UploadDocument:
    return UploadDocument(uow, storage, max_bytes=1024, max_attempts=3)


def _processor(
    uow: UnitOfWorkFactory, storage: InMemoryFileStorage, embedder: FakeEmbedder
) -> ProcessNextIngestionJob:
    return ProcessNextIngestionJob(
        uow,
        storage,
        DefaultParserRegistry(),
        StructureAwareChunker(WordCounter(), target_tokens=100, overlap_tokens=10),
        embedder,
        lease=timedelta(minutes=15),
    )


# --- Upload -------------------------------------------------------------------
async def test_upload_stores_file_and_queues_job(
    uow: UnitOfWorkFactory,
    storage: InMemoryFileStorage,
    store: InMemoryStore,
    collection: Collection,
) -> None:
    uploaded = await _upload(uow, storage).execute(
        UploadDocumentCommand(collection.id, "../../etc/k8s guide.md", GUIDE)
    )

    doc = uploaded.document
    assert doc.source_filename == "k8s guide.md"  # client path stripped
    assert doc.storage_key == f"{collection.id}/{doc.id}/k8s_guide.md"
    assert storage.files[doc.storage_key] == GUIDE
    assert doc.status is DocumentStatus.PENDING
    assert doc.title == "k8s guide"  # provisional, from the filename
    assert store.ingestion_jobs[uploaded.job.id].status is JobStatus.QUEUED


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("notes.docx", b"x", "Unsupported file type"),
        ("empty.md", b"", "empty"),
        ("fake.pdf", b"not a pdf", "not a PDF"),
        ("latin1.txt", "café".encode("latin-1"), "UTF-8"),
        ("big.md", b"x" * 2048, "upload limit"),
    ],
)
async def test_upload_rejects_invalid_files(
    uow: UnitOfWorkFactory,
    storage: InMemoryFileStorage,
    collection: Collection,
    filename: str,
    content: bytes,
    message: str,
) -> None:
    with pytest.raises(DomainValidationError, match=message):
        await _upload(uow, storage).execute(UploadDocumentCommand(collection.id, filename, content))
    assert storage.files == {}


async def test_upload_rejects_duplicate_content(
    uow: UnitOfWorkFactory, storage: InMemoryFileStorage, collection: Collection
) -> None:
    first = await _upload(uow, storage).execute(UploadDocumentCommand(collection.id, "a.md", GUIDE))

    with pytest.raises(DuplicateDocumentError) as error:
        await _upload(uow, storage).execute(UploadDocumentCommand(collection.id, "b.md", GUIDE))
    assert error.value.existing_document_id == first.document.id


async def test_upload_to_unknown_collection(
    uow: UnitOfWorkFactory, storage: InMemoryFileStorage
) -> None:
    with pytest.raises(CollectionNotFoundError):
        await _upload(uow, storage).execute(UploadDocumentCommand(new_id(), "a.md", GUIDE))


# --- Processing ---------------------------------------------------------------
async def test_process_job_end_to_end(
    uow: UnitOfWorkFactory,
    storage: InMemoryFileStorage,
    store: InMemoryStore,
    embedder: FakeEmbedder,
    collection: Collection,
) -> None:
    uploaded = await _upload(uow, storage).execute(
        UploadDocumentCommand(collection.id, "guide.md", GUIDE)
    )

    outcome = await _processor(uow, storage, embedder).execute("worker-1")

    assert outcome is not None and outcome.succeeded
    document = store.documents[uploaded.document.id]
    assert document.status is DocumentStatus.READY
    assert document.title == "Kubernetes Guide"  # discovered from the H1
    assert document.chunk_count == 2
    chunks = store.chunks[document.id]
    assert [c.heading_path for c in chunks] == [("Services",), ("Ingress",)]
    assert chunks[0].contextual_text.startswith("Document: Kubernetes Guide\nSection: Services")
    # What was embedded is the contextual text, not the bare text.
    assert embedder.document_calls == [[c.contextual_text for c in chunks]]
    assert store.ingestion_jobs[uploaded.job.id].status is JobStatus.SUCCEEDED


async def test_user_supplied_title_is_kept(
    uow: UnitOfWorkFactory,
    storage: InMemoryFileStorage,
    store: InMemoryStore,
    embedder: FakeEmbedder,
    collection: Collection,
) -> None:
    uploaded = await _upload(uow, storage).execute(
        UploadDocumentCommand(collection.id, "guide.md", GUIDE, title="My K8s Notes")
    )
    await _processor(uow, storage, embedder).execute("worker-1")

    assert store.documents[uploaded.document.id].title == "My K8s Notes"
    assert store.chunks[uploaded.document.id][0].contextual_text.startswith(
        "Document: My K8s Notes"
    )


async def test_transient_failure_is_retried_with_backoff(
    uow: UnitOfWorkFactory,
    storage: InMemoryFileStorage,
    store: InMemoryStore,
    collection: Collection,
) -> None:
    flaky = FakeEmbedder(SPEC, fail_times=1)
    uploaded = await _upload(uow, storage).execute(
        UploadDocumentCommand(collection.id, "guide.md", GUIDE)
    )

    outcome = await _processor(uow, storage, flaky).execute("worker-1")

    assert outcome is not None and outcome.will_retry
    job = store.ingestion_jobs[uploaded.job.id]
    assert job.status is JobStatus.QUEUED
    assert job.run_after > datetime.now(UTC)  # backoff: not immediately runnable
    assert store.documents[uploaded.document.id].status is DocumentStatus.PENDING
    assert await _processor(uow, storage, flaky).execute("worker-1") is None  # not due yet

    job.run_after = datetime.now(UTC)  # fast-forward past the backoff
    retried = await _processor(uow, storage, flaky).execute("worker-1")
    assert retried is not None and retried.succeeded


async def test_embedding_model_mismatch_fails_permanently(
    uow: UnitOfWorkFactory,
    storage: InMemoryFileStorage,
    store: InMemoryStore,
    collection: Collection,
) -> None:
    other_model = FakeEmbedder(EmbeddingSpec(model="other-model", dimensions=8))
    uploaded = await _upload(uow, storage).execute(
        UploadDocumentCommand(collection.id, "guide.md", GUIDE)
    )

    outcome = await _processor(uow, storage, other_model).execute("worker-1")

    assert outcome is not None and not outcome.will_retry
    assert "other-model" in (outcome.error or "")
    assert store.documents[uploaded.document.id].status is DocumentStatus.FAILED
    assert store.ingestion_jobs[uploaded.job.id].status is JobStatus.FAILED


async def test_document_without_text_fails_permanently(
    uow: UnitOfWorkFactory,
    storage: InMemoryFileStorage,
    store: InMemoryStore,
    embedder: FakeEmbedder,
    collection: Collection,
) -> None:
    uploaded = await _upload(uow, storage).execute(
        UploadDocumentCommand(collection.id, "headings.md", b"# Only\n\n## Headings\n")
    )

    outcome = await _processor(uow, storage, embedder).execute("worker-1")

    assert outcome is not None and not outcome.will_retry
    assert store.documents[uploaded.document.id].error is not None


async def test_no_work_returns_none(
    uow: UnitOfWorkFactory, storage: InMemoryFileStorage, embedder: FakeEmbedder
) -> None:
    assert await _processor(uow, storage, embedder).execute("worker-1") is None


# --- Job lifecycle ------------------------------------------------------------
def _running(job: IngestionJob) -> IngestionJob:
    job.status, job.attempts = JobStatus.RUNNING, job.attempts + 1
    return job


def test_job_backoff_doubles_and_stops_after_max_attempts() -> None:
    job = IngestionJob.for_document(new_id(), max_attempts=3)
    delays = []
    for _ in range(2):
        before = datetime.now(UTC)
        assert _running(job).fail("boom") is True
        delays.append(round((job.run_after - before).total_seconds()))
    assert delays == [30, 60]

    assert _running(job).fail("boom") is False
    assert job.status is JobStatus.FAILED
    assert job.finished_at is not None


def test_non_retryable_failure_is_final_immediately() -> None:
    job = _running(IngestionJob.for_document(new_id()))
    assert job.fail("corrupt file", retryable=False) is False
    assert job.status is JobStatus.FAILED
