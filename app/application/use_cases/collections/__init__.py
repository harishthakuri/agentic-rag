"""Collection use cases."""

from dataclasses import dataclass
from uuid import UUID

from app.application.dto.pagination import PageRequest
from app.application.ports.storage import FileStorage
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.domain.exceptions import ConflictError, NotFoundError
from app.domain.models import Collection
from app.domain.value_objects import CollectionName, EmbeddingSpec


class CollectionNotFoundError(NotFoundError):
    def __init__(self, collection_id: UUID) -> None:
        super().__init__(f"Collection {collection_id} not found")


class CollectionAlreadyExistsError(ConflictError):
    def __init__(self, name: CollectionName) -> None:
        super().__init__(f"A collection named '{name}' already exists")


@dataclass(frozen=True, slots=True)
class CreateCollectionCommand:
    name: str
    description: str | None = None


class CreateCollection:
    """New collections are bound to the currently configured embedding model."""

    def __init__(self, uow_factory: UnitOfWorkFactory, embedding: EmbeddingSpec) -> None:
        self._uow_factory = uow_factory
        self._embedding = embedding

    async def execute(self, command: CreateCollectionCommand) -> Collection:
        name = CollectionName.parse(command.name)
        collection = Collection.create(
            name=name, description=command.description, embedding=self._embedding
        )
        async with self._uow_factory() as uow:
            if await uow.collections.get_by_name(name):
                raise CollectionAlreadyExistsError(name)
            await uow.collections.add(collection)  # also raises on a concurrent duplicate
            await uow.commit()
        return collection


class GetCollection:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, collection_id: UUID) -> Collection:
        async with self._uow_factory() as uow:
            collection = await uow.collections.get(collection_id)
        if collection is None:
            raise CollectionNotFoundError(collection_id)
        return collection


class ListCollections:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, page: PageRequest) -> list[Collection]:
        async with self._uow_factory() as uow:
            return await uow.collections.list(limit=page.limit, offset=page.offset)


class DeleteCollection:
    """Deletes the collection with all its documents, chunks and jobs (DB cascade),
    then its stored files. Files go last: a leftover file is harmless, a document
    row pointing at a missing file is not."""

    def __init__(self, uow_factory: UnitOfWorkFactory, storage: FileStorage) -> None:
        self._uow_factory = uow_factory
        self._storage = storage

    async def execute(self, collection_id: UUID) -> None:
        async with self._uow_factory() as uow:
            if await uow.collections.get(collection_id) is None:
                raise CollectionNotFoundError(collection_id)
            await uow.collections.delete(collection_id)
            await uow.commit()
        await self._storage.delete_prefix(f"{collection_id}/")


__all__ = [
    "CollectionAlreadyExistsError",
    "CollectionNotFoundError",
    "CreateCollection",
    "CreateCollectionCommand",
    "DeleteCollection",
    "GetCollection",
    "ListCollections",
]
