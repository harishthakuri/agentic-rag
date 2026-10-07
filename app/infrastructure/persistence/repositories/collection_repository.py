from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.use_cases.collections import CollectionAlreadyExistsError
from app.domain.models import Collection
from app.domain.repositories import CollectionRepository
from app.domain.value_objects import CollectionName, EmbeddingSpec
from app.infrastructure.persistence.errors import is_unique_violation
from app.infrastructure.persistence.orm import CollectionEntity


class SqlAlchemyCollectionRepository(CollectionRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, collection: Collection) -> None:
        self._session.add(_to_entity(collection))
        try:
            await self._session.flush()  # surface constraint violations now, not at commit
        except IntegrityError as exc:
            if is_unique_violation(exc):
                raise CollectionAlreadyExistsError(collection.name) from exc
            raise

    async def get(self, collection_id: UUID) -> Collection | None:
        entity = await self._session.get(CollectionEntity, collection_id)
        return _to_domain(entity) if entity else None

    async def get_by_name(self, name: CollectionName) -> Collection | None:
        entity = await self._session.scalar(
            select(CollectionEntity).where(CollectionEntity.name == name.value)
        )
        return _to_domain(entity) if entity else None

    async def list(self, *, limit: int, offset: int) -> list[Collection]:
        entities = await self._session.scalars(
            select(CollectionEntity).order_by(CollectionEntity.name).limit(limit).offset(offset)
        )
        return [_to_domain(e) for e in entities]

    async def delete(self, collection_id: UUID) -> None:
        await self._session.execute(
            delete(CollectionEntity).where(CollectionEntity.id == collection_id)
        )


def _to_domain(entity: CollectionEntity) -> Collection:
    return Collection(
        id=entity.id,
        name=CollectionName(entity.name),
        description=entity.description,
        embedding=EmbeddingSpec(model=entity.embedding_model, dimensions=entity.embedding_dim),
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


def _to_entity(collection: Collection) -> CollectionEntity:
    return CollectionEntity(
        id=collection.id,
        name=collection.name.value,
        description=collection.description,
        embedding_model=collection.embedding.model,
        embedding_dim=collection.embedding.dimensions,
        created_at=collection.created_at,
        updated_at=collection.updated_at,
    )
