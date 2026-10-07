from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from app.domain.exceptions import DomainValidationError, InvalidStateTransitionError
from app.domain.models.entity import Entity
from app.domain.value_objects import ContentHash, new_id


class DocumentType(StrEnum):
    """Supported source formats, keyed by MIME type."""

    MARKDOWN = "text/markdown"
    PLAIN_TEXT = "text/plain"
    PDF = "application/pdf"

    @classmethod
    def from_filename(cls, filename: str) -> "DocumentType":
        """Decide by extension; clients' Content-Type headers are often generic or wrong."""
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if extension not in _EXTENSIONS:
            supported = ", ".join(f".{ext}" for ext in sorted(_EXTENSIONS))
            raise DomainValidationError(f"Unsupported file type. Supported: {supported}")
        return _EXTENSIONS[extension]


_EXTENSIONS = {
    "md": DocumentType.MARKDOWN,
    "markdown": DocumentType.MARKDOWN,
    "txt": DocumentType.PLAIN_TEXT,
    "pdf": DocumentType.PDF,
}


class DocumentStatus(StrEnum):
    PENDING = "pending"  # uploaded, waiting for the ingestion worker
    PROCESSING = "processing"  # being parsed, chunked and embedded
    READY = "ready"  # searchable
    FAILED = "failed"  # ingestion gave up; see `error`


_ALLOWED_TRANSITIONS: dict[DocumentStatus, frozenset[DocumentStatus]] = {
    DocumentStatus.PENDING: frozenset({DocumentStatus.PROCESSING}),
    DocumentStatus.PROCESSING: frozenset(
        {DocumentStatus.READY, DocumentStatus.FAILED, DocumentStatus.PENDING}  # PENDING = retry
    ),
    DocumentStatus.READY: frozenset({DocumentStatus.PENDING}),  # re-ingest
    DocumentStatus.FAILED: frozenset({DocumentStatus.PENDING}),  # manual retry
}


@dataclass(eq=False, kw_only=True, slots=True)
class Document(Entity):
    """A source file in a collection, and where it is in the ingestion lifecycle."""

    id: UUID = field(default_factory=new_id)
    collection_id: UUID
    title: str
    source_filename: str
    document_type: DocumentType
    content_hash: ContentHash
    size_bytes: int
    storage_key: str
    status: DocumentStatus = DocumentStatus.PENDING
    error: str | None = None
    chunk_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise DomainValidationError("Document title must not be empty")
        if self.size_bytes < 0:
            raise DomainValidationError("Document size must not be negative")

    # --- Lifecycle ---------------------------------------------------------
    def start_processing(self) -> None:
        self._transition(DocumentStatus.PROCESSING)
        self.error = None

    def mark_ready(self, chunk_count: int) -> None:
        if chunk_count < 0:
            raise DomainValidationError("Chunk count must not be negative")
        self._transition(DocumentStatus.READY)
        self.chunk_count = chunk_count

    def mark_failed(self, error: str) -> None:
        self._transition(DocumentStatus.FAILED)
        self.error = error

    def requeue(self) -> None:
        self._transition(DocumentStatus.PENDING)

    def retitle(self, title: str) -> None:
        if not title.strip():
            raise DomainValidationError("Document title must not be empty")
        self.title = title.strip()
        self.updated_at = datetime.now(UTC)

    def _transition(self, target: DocumentStatus) -> None:
        if target not in _ALLOWED_TRANSITIONS[self.status]:
            raise InvalidStateTransitionError(
                f"Document {self.id} cannot move from '{self.status}' to '{target}'"
            )
        self.status = target
        self.updated_at = datetime.now(UTC)
