from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from app.domain.models import Document, DocumentStatus


class DocumentResponse(BaseModel):
    id: UUID
    collection_id: UUID
    title: str
    source_filename: str
    mime_type: str
    size_bytes: int
    content_hash: str
    status: DocumentStatus
    error: str | None
    chunk_count: int
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, document: Document) -> "DocumentResponse":
        return cls(
            id=document.id,
            collection_id=document.collection_id,
            title=document.title,
            source_filename=document.source_filename,
            mime_type=document.document_type.value,
            size_bytes=document.size_bytes,
            content_hash=document.content_hash.value,
            status=document.status,
            error=document.error,
            chunk_count=document.chunk_count,
            metadata=document.metadata,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )
