"""In-memory implementations of the persistence ports, for fast use-case tests.

They honour the same contract as the SQLAlchemy versions (including conflict
errors and commit/rollback semantics), which is what makes the ports useful.
"""

import copy
import hashlib
import math
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Self
from uuid import UUID

from app.application.ports.chat import (
    ChatMessage,
    ChatModelError,
    ChatStreamEvent,
    CompletionDone,
    TextDelta,
    TokenUsage,
)
from app.application.ports.embeddings import EmbeddingUnavailableError
from app.application.ports.reranking import RerankCandidate, RerankError, RerankScore
from app.application.ports.search import ChunkMatch
from app.application.ports.unit_of_work import UnitOfWork
from app.application.use_cases.collections import CollectionAlreadyExistsError
from app.application.use_cases.documents import DuplicateDocumentError
from app.domain.models import ApiKey, Chunk, Collection, Document, IngestionJob, JobStatus
from app.domain.repositories import (
    ApiKeyRepository,
    ChunkRepository,
    CollectionRepository,
    DocumentRepository,
    IngestionJobRepository,
)
from app.domain.value_objects import CollectionName, ContentHash, EmbeddingSpec


class InMemoryCollectionRepository(CollectionRepository):
    def __init__(self, rows: dict[UUID, Collection]) -> None:
        self.rows = rows

    async def add(self, collection: Collection) -> None:
        if any(c.name == collection.name for c in self.rows.values()):
            raise CollectionAlreadyExistsError(collection.name)
        self.rows[collection.id] = collection

    async def get(self, collection_id: UUID) -> Collection | None:
        return self.rows.get(collection_id)

    async def get_by_name(self, name: CollectionName) -> Collection | None:
        return next((c for c in self.rows.values() if c.name == name), None)

    async def list(self, *, limit: int, offset: int) -> list[Collection]:
        return sorted(self.rows.values(), key=lambda c: c.name.value)[offset : offset + limit]

    async def delete(self, collection_id: UUID) -> None:
        self.rows.pop(collection_id, None)


class InMemoryDocumentRepository(DocumentRepository):
    def __init__(self, rows: dict[UUID, Document]) -> None:
        self.rows = rows

    async def add(self, document: Document) -> None:
        if await self.get_by_content_hash(document.collection_id, document.content_hash):
            raise DuplicateDocumentError
        self.rows[document.id] = document

    async def update(self, document: Document) -> None:
        self.rows[document.id] = document

    async def get(self, document_id: UUID) -> Document | None:
        return self.rows.get(document_id)

    async def get_by_content_hash(
        self, collection_id: UUID, content_hash: ContentHash
    ) -> Document | None:
        return next(
            (
                d
                for d in self.rows.values()
                if d.collection_id == collection_id and d.content_hash == content_hash
            ),
            None,
        )

    async def list_by_collection(
        self, collection_id: UUID, *, limit: int, offset: int
    ) -> list[Document]:
        docs = [d for d in self.rows.values() if d.collection_id == collection_id]
        return sorted(docs, key=lambda d: d.created_at, reverse=True)[offset : offset + limit]

    async def delete(self, document_id: UUID) -> None:
        self.rows.pop(document_id, None)


class InMemoryApiKeyRepository(ApiKeyRepository):
    def __init__(self, rows: dict[UUID, ApiKey]) -> None:
        self.rows = rows

    async def add(self, api_key: ApiKey) -> None:
        self.rows[api_key.id] = api_key

    async def update(self, api_key: ApiKey) -> None:
        self.rows[api_key.id] = api_key

    async def get(self, api_key_id: UUID) -> ApiKey | None:
        return self.rows.get(api_key_id)

    async def get_by_hash(self, key_hash: str) -> ApiKey | None:
        return next((k for k in self.rows.values() if k.key_hash == key_hash), None)

    async def list(self) -> list[ApiKey]:
        return list(self.rows.values())


class InMemoryChunkRepository(ChunkRepository):
    def __init__(self, rows: dict[UUID, list[Chunk]]) -> None:
        self.rows = rows

    async def replace_for_document(self, document_id: UUID, chunks: Sequence[Chunk]) -> None:
        self.rows[document_id] = list(chunks)

    async def count_for_document(self, document_id: UUID) -> int:
        return len(self.rows.get(document_id, []))


class InMemoryIngestionJobRepository(IngestionJobRepository):
    def __init__(self, rows: dict[UUID, IngestionJob]) -> None:
        self.rows = rows

    async def add(self, job: IngestionJob) -> None:
        self.rows[job.id] = job

    async def update(self, job: IngestionJob) -> None:
        self.rows[job.id] = job

    async def get(self, job_id: UUID) -> IngestionJob | None:
        return self.rows.get(job_id)

    async def claim_next(self, worker_id: str, lease: timedelta) -> IngestionJob | None:
        now = datetime.now(UTC)
        due = [
            j
            for j in self.rows.values()
            if (j.status is JobStatus.QUEUED and j.run_after <= now)
            or (
                j.status is JobStatus.RUNNING
                and j.locked_at is not None
                and j.locked_at < now - lease
                and j.attempts < j.max_attempts
            )
        ]
        if not due:
            return None
        job = min(due, key=lambda j: j.run_after)
        job.status, job.attempts = JobStatus.RUNNING, job.attempts + 1
        job.locked_at, job.locked_by = now, worker_id
        return job


class InMemoryChunkSearchIndex:
    """Naive versions of the two retrievers: exact cosine similarity, and the number
    of shared lowercase words standing in for full-text ranking."""

    def __init__(self, store: "InMemoryStore") -> None:
        self._store = store

    def _chunks(self, collection_id: UUID) -> list[Chunk]:
        return [
            c
            for chunks in self._store.chunks.values()
            for c in chunks
            if c.collection_id == collection_id
        ]

    def _match(self, chunk: Chunk, score: float) -> ChunkMatch:
        document = self._store.documents[chunk.document_id]
        return ChunkMatch(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_title=document.title,
            ordinal=chunk.ordinal,
            text=chunk.text,
            heading_path=chunk.heading_path,
            page=chunk.page,
            score=score,
        )

    async def vector_search(
        self, collection_id: UUID, embedding: Sequence[float], limit: int
    ) -> list[ChunkMatch]:
        scored = [
            (sum(a * b for a, b in zip(c.embedding, embedding, strict=True)), c)
            for c in self._chunks(collection_id)
        ]
        scored.sort(key=lambda pair: -pair[0])
        return [self._match(c, s) for s, c in scored[:limit]]

    async def keyword_search(self, collection_id: UUID, query: str, limit: int) -> list[ChunkMatch]:
        terms = set(query.lower().split())
        scored = [
            (len(terms & set(c.contextual_text.lower().split())), c)
            for c in self._chunks(collection_id)
        ]
        scored = [pair for pair in scored if pair[0] > 0]
        scored.sort(key=lambda pair: -pair[0])
        return [self._match(c, float(s)) for s, c in scored[:limit]]


class InMemoryStore:
    """The 'database': committed state shared by every unit of work."""

    def __init__(self) -> None:
        self.collections: dict[UUID, Collection] = {}
        self.documents: dict[UUID, Document] = {}
        self.chunks: dict[UUID, list[Chunk]] = {}
        self.ingestion_jobs: dict[UUID, IngestionJob] = {}
        self.api_keys: dict[UUID, ApiKey] = {}
        self.commits = 0


_TABLES = ("collections", "documents", "chunks", "ingestion_jobs", "api_keys")


class InMemoryUnitOfWork(UnitOfWork):
    """Works on a copy of the store; `commit` publishes it, anything else discards it."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    async def __aenter__(self) -> Self:
        self._tables = {name: copy.deepcopy(getattr(self._store, name)) for name in _TABLES}
        self.collections = InMemoryCollectionRepository(self._tables["collections"])
        self.documents = InMemoryDocumentRepository(self._tables["documents"])
        self.chunks = InMemoryChunkRepository(self._tables["chunks"])
        self.ingestion_jobs = InMemoryIngestionJobRepository(self._tables["ingestion_jobs"])
        self.api_keys = InMemoryApiKeyRepository(self._tables["api_keys"])
        self.search = InMemoryChunkSearchIndex(self._store)  # reads committed state
        return await super().__aenter__()

    async def commit(self) -> None:
        for name, rows in self._tables.items():
            setattr(self._store, name, copy.deepcopy(rows))
        self._store.commits += 1

    async def rollback(self) -> None:
        pass  # uncommitted copies are simply dropped


class InMemoryFileStorage:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def save(self, key: str, content: bytes) -> None:
        self.files[key] = content

    async def read(self, key: str) -> bytes:
        return self.files[key]

    async def delete_prefix(self, prefix: str) -> None:
        for key in [k for k in self.files if k.startswith(prefix)]:
            del self.files[key]


class FakeEmbedder:
    """Deterministic vectors derived from a hash of the text: identical texts get
    identical vectors, different texts different ones. No model required."""

    def __init__(self, spec: EmbeddingSpec, *, fail_times: int = 0) -> None:
        self._spec = spec
        self.fail_times = fail_times
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise EmbeddingUnavailableError("simulated outage")
        self.document_calls.append(list(texts))
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode()).digest()
        raw = [seed[i % len(seed)] - 127.5 for i in range(self._spec.dimensions)]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]


class FakeReranker:
    """Grades a candidate by how many of `preferred` words its text contains."""

    def __init__(self, preferred: set[str] | None = None, *, fail: bool = False) -> None:
        self._preferred = preferred or set()
        self._fail = fail
        self.calls: list[tuple[str, list[RerankCandidate]]] = []

    @property
    def name(self) -> str:
        return "fake"

    async def rerank(self, query: str, candidates: Sequence[RerankCandidate]) -> list[RerankScore]:
        self.calls.append((query, list(candidates)))
        if self._fail:
            raise RerankError("simulated reranker outage")
        return [
            RerankScore(id=c.id, score=float(len(self._preferred & set(c.text.split()))))
            for c in candidates
        ]


class FakeChatModel:
    """Streams a scripted reply in small pieces and records the prompts it received."""

    def __init__(
        self, reply: str | Callable[[Sequence[ChatMessage]], str] = "", *, fail: bool = False
    ) -> None:
        self._reply = reply
        self._fail = fail
        self.calls: list[list[ChatMessage]] = []

    @property
    def model(self) -> str:
        return "fake-chat"

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[ChatStreamEvent]:
        self.calls.append(list(messages))
        if self._fail:
            raise ChatModelError("simulated outage")
        text = self._reply(messages) if callable(self._reply) else self._reply
        for start in range(0, len(text), 4):
            yield TextDelta(text[start : start + 4])
        yield CompletionDone(TokenUsage(prompt_tokens=100, completion_tokens=len(text) // 4))
