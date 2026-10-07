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
    ToolCall,
    ToolSpec,
)
from app.application.ports.embeddings import EmbeddingUnavailableError
from app.application.ports.reranking import RerankCandidate, RerankError, RerankScore
from app.application.ports.search import ChunkMatch
from app.application.ports.unit_of_work import UnitOfWork
from app.application.use_cases.collections import CollectionAlreadyExistsError
from app.application.use_cases.documents import DuplicateDocumentError
from app.domain.models import (
    AgentRun,
    AgentStep,
    ApiKey,
    Chunk,
    Collection,
    Document,
    IngestionJob,
    JobStatus,
)
from app.domain.repositories import (
    AgentRunRepository,
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


class InMemoryAgentRunRepository(AgentRunRepository):
    def __init__(self, runs: dict[UUID, AgentRun], steps: dict[UUID, list[AgentStep]]) -> None:
        self.runs = runs
        self.step_rows = steps

    async def add(self, run: AgentRun) -> None:
        self.runs[run.id] = run

    async def update(self, run: AgentRun) -> None:
        self.runs[run.id] = run

    async def get(self, run_id: UUID) -> AgentRun | None:
        return self.runs.get(run_id)

    async def add_step(self, step: AgentStep) -> None:
        self.step_rows.setdefault(step.run_id, []).append(step)

    async def steps(self, run_id: UUID) -> list[AgentStep]:
        return sorted(self.step_rows.get(run_id, []), key=lambda s: s.number)


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

    async def neighbours(
        self, collection_id: UUID, chunk_id: UUID, before: int, after: int
    ) -> list[ChunkMatch]:
        chunks = self._chunks(collection_id)
        target = next((c for c in chunks if c.id == chunk_id), None)
        if target is None:
            return []
        lo, hi = target.ordinal - before, target.ordinal + after
        return [
            self._match(c, 0.0)
            for c in sorted(chunks, key=lambda c: c.ordinal)
            if c.document_id == target.document_id and lo <= c.ordinal <= hi
        ]


class InMemoryStore:
    """The 'database': committed state shared by every unit of work."""

    def __init__(self) -> None:
        self.collections: dict[UUID, Collection] = {}
        self.documents: dict[UUID, Document] = {}
        self.chunks: dict[UUID, list[Chunk]] = {}
        self.ingestion_jobs: dict[UUID, IngestionJob] = {}
        self.api_keys: dict[UUID, ApiKey] = {}
        self.agent_runs: dict[UUID, AgentRun] = {}
        self.agent_steps: dict[UUID, list[AgentStep]] = {}
        self.commits = 0


_TABLES = (
    "collections",
    "documents",
    "chunks",
    "ingestion_jobs",
    "api_keys",
    "agent_runs",
    "agent_steps",
)


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
        self.agent_runs = InMemoryAgentRunRepository(
            self._tables["agent_runs"], self._tables["agent_steps"]
        )
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


type FakeTurn = str | list[ToolCall]
type FakeReply = str | list[FakeTurn] | Callable[[Sequence[ChatMessage]], FakeTurn]


class FakeChatModel:
    """Streams scripted turns and records every request it received.

    `reply` is either one answer used for every turn, a callable producing a turn
    from the messages, or a list of turns consumed in order. A turn is answer text
    (str) or a list of tool calls.
    """

    def __init__(
        self,
        reply: FakeReply = "",
        *,
        fail: bool = False,
    ) -> None:
        self._reply = reply
        self._fail = fail
        self.calls: list[list[ChatMessage]] = []
        self.tools_offered: list[list[str]] = []

    @property
    def model(self) -> str:
        return "fake-chat"

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[ChatStreamEvent]:
        self.calls.append(list(messages))
        self.tools_offered.append([t.name for t in tools])
        if self._fail:
            raise ChatModelError("simulated outage")
        turn = self._next_turn(messages)
        usage = TokenUsage(prompt_tokens=100, completion_tokens=10)
        if isinstance(turn, list):
            yield CompletionDone(usage, tool_calls=tuple(turn))
            return
        for start in range(0, len(turn), 4):
            yield TextDelta(turn[start : start + 4])
        yield CompletionDone(usage)

    def _next_turn(self, messages: Sequence[ChatMessage]) -> FakeTurn:
        if isinstance(self._reply, str):
            return self._reply
        if isinstance(self._reply, list):
            return self._reply.pop(0) if self._reply else "(script exhausted)"
        return self._reply(messages)
