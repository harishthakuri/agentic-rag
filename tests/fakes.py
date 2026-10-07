"""In-memory implementations of the persistence ports, for fast use-case tests.

They honour the same contract as the SQLAlchemy versions (including conflict
errors and commit/rollback semantics), which is what makes the ports useful.
"""

import copy
from typing import Self
from uuid import UUID

from app.application.ports.unit_of_work import UnitOfWork
from app.application.use_cases.collections import CollectionAlreadyExistsError
from app.application.use_cases.documents import DuplicateDocumentError
from app.domain.models import ApiKey, Collection, Document
from app.domain.repositories import ApiKeyRepository, CollectionRepository, DocumentRepository
from app.domain.value_objects import CollectionName, ContentHash


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


class InMemoryStore:
    """The 'database': committed state shared by every unit of work."""

    def __init__(self) -> None:
        self.collections: dict[UUID, Collection] = {}
        self.documents: dict[UUID, Document] = {}
        self.api_keys: dict[UUID, ApiKey] = {}
        self.commits = 0


class InMemoryUnitOfWork(UnitOfWork):
    """Works on a copy of the store; `commit` publishes it, anything else discards it."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    async def __aenter__(self) -> Self:
        self._collections = copy.deepcopy(self._store.collections)
        self._documents = copy.deepcopy(self._store.documents)
        self._api_keys = copy.deepcopy(self._store.api_keys)
        self.collections = InMemoryCollectionRepository(self._collections)
        self.documents = InMemoryDocumentRepository(self._documents)
        self.api_keys = InMemoryApiKeyRepository(self._api_keys)
        return await super().__aenter__()

    async def commit(self) -> None:
        self._store.collections = copy.deepcopy(self._collections)
        self._store.documents = copy.deepcopy(self._documents)
        self._store.api_keys = copy.deepcopy(self._api_keys)
        self._store.commits += 1

    async def rollback(self) -> None:
        pass  # uncommitted copies are simply dropped
