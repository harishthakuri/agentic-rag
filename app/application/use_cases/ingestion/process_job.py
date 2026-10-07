"""The ingestion pipeline, run by the worker for one job at a time:

    claim job → read file → parse → chunk → embed → store chunks → document READY

Database transactions are kept short on purpose. The slow middle part (parsing,
and especially embedding) runs with no transaction open, so no locks or
connections are held while the embedding model works.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID

from app.application.ports.chunking import ChunkDraft, Chunker
from app.application.ports.embeddings import EmbeddingProvider
from app.application.ports.parsing import ParserRegistry, UnparseableDocumentError
from app.application.ports.storage import FileStorage
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.domain.exceptions import DomainError
from app.domain.models import Chunk, Collection, Document, DocumentStatus, IngestionJob

logger = logging.getLogger(__name__)

_MAX_ERROR_LENGTH = 1000


class PermanentIngestionError(Exception):
    """Retrying will not help (bad file, wrong embedding model, empty document)."""


@dataclass(frozen=True, slots=True)
class JobOutcome:
    job_id: UUID
    document_id: UUID
    succeeded: bool
    chunk_count: int = 0
    will_retry: bool = False
    error: str | None = None


class ProcessNextIngestionJob:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        storage: FileStorage,
        parsers: ParserRegistry,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        *,
        lease: timedelta,
    ) -> None:
        self._uow_factory = uow_factory
        self._storage = storage
        self._parsers = parsers
        self._chunker = chunker
        self._embedder = embedder
        self._lease = lease

    async def execute(self, worker_id: str) -> JobOutcome | None:
        """Process one job. Returns None when there is nothing to do."""
        claimed = await self._claim(worker_id)
        if claimed is None:
            return None
        job, document, collection = claimed

        try:
            chunks, discovered_title = await self._build_chunks(document, collection)
        except Exception as exc:
            return await self._record_failure(job.id, document.id, exc)

        return await self._store(job.id, document.id, chunks, discovered_title)

    # --- Step 1: claim (short transaction) --------------------------------
    async def _claim(self, worker_id: str) -> tuple[IngestionJob, Document, Collection] | None:
        async with self._uow_factory() as uow:
            job = await uow.ingestion_jobs.claim_next(worker_id, self._lease)
            if job is None:
                return None
            document = await uow.documents.get(job.document_id)
            collection = await uow.collections.get(document.collection_id) if document else None
            if document is None or collection is None:  # deleted meanwhile; cascade removes job
                await uow.commit()
                return None
            if document.status is not DocumentStatus.PROCESSING:  # PROCESSING = lease reclaim
                document.start_processing()
                await uow.documents.update(document)
            await uow.commit()
        return job, document, collection

    # --- Step 2: the slow part (no transaction) ----------------------------
    async def _build_chunks(
        self, document: Document, collection: Collection
    ) -> tuple[list[Chunk], str | None]:
        """Returns the chunks and, if the user gave no title, one found in the document."""
        if collection.embedding != self._embedder.spec:
            raise PermanentIngestionError(
                f"Collection uses {collection.embedding.model} ({collection.embedding.dimensions}"
                f" dims) but the configured embedder is {self._embedder.spec.model} "
                f"({self._embedder.spec.dimensions} dims)"
            )

        content = await self._storage.read(document.storage_key)
        try:
            parsed = self._parsers.for_type(document.document_type).parse(content)
        except UnparseableDocumentError as exc:
            raise PermanentIngestionError(f"Could not parse the file: {exc}") from exc

        discovered_title = (
            parsed.title if document.metadata.get("title_source") == "filename" else None
        )
        drafts = self._chunker.chunk(parsed, title=discovered_title or document.title)
        if not drafts:
            raise PermanentIngestionError("The document contains no extractable text")

        # The contextual text (title + headings + body) is what gets embedded.
        vectors = await self._embedder.embed_documents([d.contextual_text for d in drafts])
        return _to_chunks(document, drafts, vectors), discovered_title

    # --- Step 3: store (short transaction) ---------------------------------
    async def _store(
        self, job_id: UUID, document_id: UUID, chunks: list[Chunk], title: str | None
    ) -> JobOutcome:
        async with self._uow_factory() as uow:
            job = await uow.ingestion_jobs.get(job_id)
            document = await uow.documents.get(document_id)
            if job is None or document is None:  # deleted while we were embedding
                return JobOutcome(job_id, document_id, succeeded=False, error="deleted")

            await uow.chunks.replace_for_document(document_id, chunks)
            if title:
                document.retitle(title)
            document.mark_ready(chunk_count=len(chunks))
            job.succeed()
            await uow.documents.update(document)
            await uow.ingestion_jobs.update(job)
            await uow.commit()
        return JobOutcome(job_id, document_id, succeeded=True, chunk_count=len(chunks))

    async def _record_failure(self, job_id: UUID, document_id: UUID, exc: Exception) -> JobOutcome:
        retryable = not isinstance(exc, PermanentIngestionError | DomainError)
        message = f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_LENGTH]
        if retryable:
            logger.warning("Ingestion attempt failed for job %s: %s", job_id, message)
        else:
            logger.info("Ingestion failed permanently for job %s: %s", job_id, message)

        async with self._uow_factory() as uow:
            job = await uow.ingestion_jobs.get(job_id)
            document = await uow.documents.get(document_id)
            if job is None or document is None:
                return JobOutcome(job_id, document_id, succeeded=False, error=message)
            will_retry = job.fail(message, retryable=retryable)
            if will_retry:
                document.requeue()
            else:
                document.mark_failed(message)
            await uow.ingestion_jobs.update(job)
            await uow.documents.update(document)
            await uow.commit()
        return JobOutcome(
            job_id, document_id, succeeded=False, will_retry=will_retry, error=message
        )


def _to_chunks(
    document: Document, drafts: Sequence[ChunkDraft], vectors: Sequence[list[float]]
) -> list[Chunk]:
    if len(vectors) != len(drafts):
        raise RuntimeError(f"Embedder returned {len(vectors)} vectors for {len(drafts)} chunks")
    return [
        Chunk(
            document_id=document.id,
            collection_id=document.collection_id,
            ordinal=ordinal,
            text=draft.text,
            contextual_text=draft.contextual_text,
            heading_path=draft.heading_path,
            page=draft.page,
            token_count=draft.token_count,
            embedding=vector,
        )
        for ordinal, (draft, vector) in enumerate(zip(drafts, vectors, strict=True))
    ]
