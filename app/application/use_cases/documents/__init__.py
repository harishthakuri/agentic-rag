"""Document read/delete use cases. Upload and ingestion arrive with the ingestion pipeline."""

from uuid import UUID

from app.application.dto.pagination import PageRequest
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.use_cases.collections import CollectionNotFoundError
from app.domain.exceptions import ConflictError, NotFoundError
from app.domain.models import Document


class DocumentNotFoundError(NotFoundError):
    def __init__(self, document_id: UUID) -> None:
        super().__init__(f"Document {document_id} not found")


class DuplicateDocumentError(ConflictError):
    def __init__(self) -> None:
        super().__init__("A document with identical content already exists in this collection")


class ListDocuments:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, collection_id: UUID, page: PageRequest) -> list[Document]:
        async with self._uow_factory() as uow:
            if await uow.collections.get(collection_id) is None:
                raise CollectionNotFoundError(collection_id)
            return await uow.documents.list_by_collection(
                collection_id, limit=page.limit, offset=page.offset
            )


class GetDocument:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, document_id: UUID) -> Document:
        async with self._uow_factory() as uow:
            document = await uow.documents.get(document_id)
        if document is None:
            raise DocumentNotFoundError(document_id)
        return document


class DeleteDocument:
    """Deletes the document with its chunks and jobs (DB cascade)."""

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, document_id: UUID) -> None:
        async with self._uow_factory() as uow:
            if await uow.documents.get(document_id) is None:
                raise DocumentNotFoundError(document_id)
            await uow.documents.delete(document_id)
            await uow.commit()


__all__ = [
    "DeleteDocument",
    "DocumentNotFoundError",
    "DuplicateDocumentError",
    "GetDocument",
    "ListDocuments",
]
