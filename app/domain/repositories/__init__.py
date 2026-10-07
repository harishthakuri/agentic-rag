"""Repository interfaces. Implemented in infrastructure; used through the Unit of Work.

Repositories never commit; the Unit of Work owns the transaction.
"""

from abc import ABC, abstractmethod
from uuid import UUID

from app.domain.models import ApiKey, Collection, Document
from app.domain.value_objects import CollectionName, ContentHash


class CollectionRepository(ABC):
    @abstractmethod
    async def add(self, collection: Collection) -> None:
        """Raises ConflictError if the name is already taken."""

    @abstractmethod
    async def get(self, collection_id: UUID) -> Collection | None: ...

    @abstractmethod
    async def get_by_name(self, name: CollectionName) -> Collection | None: ...

    @abstractmethod
    async def list(self, *, limit: int, offset: int) -> list[Collection]: ...

    @abstractmethod
    async def delete(self, collection_id: UUID) -> None:
        """Deletes the collection and, by cascade, its documents and chunks."""


class DocumentRepository(ABC):
    @abstractmethod
    async def add(self, document: Document) -> None:
        """Raises ConflictError if the same content already exists in the collection."""

    @abstractmethod
    async def update(self, document: Document) -> None: ...

    @abstractmethod
    async def get(self, document_id: UUID) -> Document | None: ...

    @abstractmethod
    async def get_by_content_hash(
        self, collection_id: UUID, content_hash: ContentHash
    ) -> Document | None: ...

    @abstractmethod
    async def list_by_collection(
        self, collection_id: UUID, *, limit: int, offset: int
    ) -> list[Document]: ...

    @abstractmethod
    async def delete(self, document_id: UUID) -> None: ...


class ApiKeyRepository(ABC):
    @abstractmethod
    async def add(self, api_key: ApiKey) -> None: ...

    @abstractmethod
    async def update(self, api_key: ApiKey) -> None: ...

    @abstractmethod
    async def get(self, api_key_id: UUID) -> ApiKey | None: ...

    @abstractmethod
    async def get_by_hash(self, key_hash: str) -> ApiKey | None: ...

    @abstractmethod
    async def list(self) -> list[ApiKey]: ...


__all__ = ["ApiKeyRepository", "CollectionRepository", "DocumentRepository"]
