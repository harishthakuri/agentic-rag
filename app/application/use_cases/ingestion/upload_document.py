"""Upload: validate and store the file, record the document, enqueue ingestion.

Returns immediately (HTTP 202). Parsing, chunking and embedding are slow, so
they happen in the worker (see ProcessNextIngestionJob).
"""

import re
from dataclasses import dataclass
from pathlib import PurePath
from uuid import UUID

from app.application.ports.storage import FileStorage
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.use_cases.collections import CollectionNotFoundError
from app.application.use_cases.documents import DuplicateDocumentError
from app.domain.exceptions import DomainValidationError
from app.domain.models import Document, DocumentType, IngestionJob
from app.domain.value_objects import ContentHash

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class UploadDocumentCommand:
    collection_id: UUID
    filename: str
    content: bytes
    title: str | None = None


@dataclass(frozen=True, slots=True)
class UploadedDocument:
    document: Document
    job: IngestionJob


class UploadDocument:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        storage: FileStorage,
        *,
        max_bytes: int,
        max_attempts: int,
    ) -> None:
        self._uow_factory = uow_factory
        self._storage = storage
        self._max_bytes = max_bytes
        self._max_attempts = max_attempts

    async def execute(self, command: UploadDocumentCommand) -> UploadedDocument:
        filename = PurePath(command.filename).name  # strip any client-supplied path
        document_type = DocumentType.from_filename(filename)
        self._validate_content(command.content, document_type)
        content_hash = ContentHash.of(command.content)

        async with self._uow_factory() as uow:
            if await uow.collections.get(command.collection_id) is None:
                raise CollectionNotFoundError(command.collection_id)
            existing = await uow.documents.get_by_content_hash(command.collection_id, content_hash)
            if existing:
                raise DuplicateDocumentError(existing.id)

        user_title = (command.title or "").strip()
        document = Document(
            collection_id=command.collection_id,
            title=user_title or _stem(filename),
            source_filename=filename,
            document_type=document_type,
            content_hash=content_hash,
            size_bytes=len(command.content),
            storage_key="",  # set below, once the id exists
            # Without an explicit title, the worker may use one found in the document.
            metadata={"title_source": "user" if user_title else "filename"},
        )
        document.storage_key = f"{command.collection_id}/{document.id}/{_safe_filename(filename)}"
        job = IngestionJob.for_document(document.id, max_attempts=self._max_attempts)

        # File first, then the rows: if the commit fails, remove the orphaned file.
        await self._storage.save(document.storage_key, command.content)
        try:
            async with self._uow_factory() as uow:
                await uow.documents.add(document)
                await uow.ingestion_jobs.add(job)
                await uow.commit()
        except BaseException:
            await self._storage.delete_prefix(f"{command.collection_id}/{document.id}/")
            raise
        return UploadedDocument(document=document, job=job)

    def _validate_content(self, content: bytes, document_type: DocumentType) -> None:
        if not content:
            raise DomainValidationError("The file is empty")
        if len(content) > self._max_bytes:
            raise DomainValidationError(
                f"The file exceeds the {self._max_bytes // (1024 * 1024)} MB upload limit"
            )
        if document_type is DocumentType.PDF:
            if not content.startswith(b"%PDF-"):
                raise DomainValidationError("The file has a .pdf extension but is not a PDF")
        else:
            try:
                content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise DomainValidationError("Text files must be UTF-8 encoded") from exc


def _stem(filename: str) -> str:
    return PurePath(filename).stem.replace("_", " ").replace("-", " ").strip() or filename


def _safe_filename(filename: str) -> str:
    return _UNSAFE_FILENAME_CHARS.sub("_", filename)[-120:] or "upload"
