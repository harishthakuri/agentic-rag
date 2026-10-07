from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.use_cases.documents import DocumentNotFoundError, DuplicateDocumentError
from app.domain.models import Document, DocumentStatus, DocumentType
from app.domain.repositories import DocumentRepository
from app.domain.value_objects import ContentHash
from app.infrastructure.persistence.errors import is_unique_violation
from app.infrastructure.persistence.orm import DocumentEntity


class SqlAlchemyDocumentRepository(DocumentRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, document: Document) -> None:
        entity = DocumentEntity(id=document.id, created_at=document.created_at)
        _apply(document, entity)
        self._session.add(entity)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            if is_unique_violation(exc):
                raise DuplicateDocumentError from exc
            raise

    async def update(self, document: Document) -> None:
        entity = await self._session.get(DocumentEntity, document.id)
        if entity is None:
            raise DocumentNotFoundError(document.id)
        _apply(document, entity)
        await self._session.flush()

    async def get(self, document_id: UUID) -> Document | None:
        entity = await self._session.get(DocumentEntity, document_id)
        return _to_domain(entity) if entity else None

    async def get_by_content_hash(
        self, collection_id: UUID, content_hash: ContentHash
    ) -> Document | None:
        entity = await self._session.scalar(
            select(DocumentEntity).where(
                DocumentEntity.collection_id == collection_id,
                DocumentEntity.content_hash == content_hash.value,
            )
        )
        return _to_domain(entity) if entity else None

    async def list_by_collection(
        self, collection_id: UUID, *, limit: int, offset: int
    ) -> list[Document]:
        entities = await self._session.scalars(
            select(DocumentEntity)
            .where(DocumentEntity.collection_id == collection_id)
            .order_by(DocumentEntity.created_at.desc(), DocumentEntity.id)
            .limit(limit)
            .offset(offset)
        )
        return [_to_domain(e) for e in entities]

    async def delete(self, document_id: UUID) -> None:
        await self._session.execute(delete(DocumentEntity).where(DocumentEntity.id == document_id))


def _to_domain(entity: DocumentEntity) -> Document:
    return Document(
        id=entity.id,
        collection_id=entity.collection_id,
        title=entity.title,
        source_filename=entity.source_filename,
        document_type=DocumentType(entity.mime_type),
        content_hash=ContentHash(entity.content_hash),
        size_bytes=entity.size_bytes,
        storage_key=entity.storage_key,
        status=DocumentStatus(entity.status),
        error=entity.error,
        chunk_count=entity.chunk_count,
        metadata=dict(entity.metadata_),
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


def _apply(document: Document, entity: DocumentEntity) -> None:
    """Copy mutable domain state onto the ORM entity."""
    entity.collection_id = document.collection_id
    entity.title = document.title
    entity.source_filename = document.source_filename
    entity.mime_type = document.document_type.value
    entity.content_hash = document.content_hash.value
    entity.size_bytes = document.size_bytes
    entity.storage_key = document.storage_key
    entity.status = document.status.value
    entity.error = document.error
    entity.chunk_count = document.chunk_count
    entity.metadata_ = dict(document.metadata)
    entity.updated_at = document.updated_at
