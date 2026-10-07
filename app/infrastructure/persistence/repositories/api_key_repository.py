from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.exceptions import NotFoundError
from app.domain.models import ApiKey
from app.domain.repositories import ApiKeyRepository
from app.infrastructure.persistence.orm import ApiKeyEntity


class SqlAlchemyApiKeyRepository(ApiKeyRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, api_key: ApiKey) -> None:
        self._session.add(
            ApiKeyEntity(
                id=api_key.id,
                name=api_key.name,
                display_prefix=api_key.display_prefix,
                key_hash=api_key.key_hash,
                created_at=api_key.created_at,
                last_used_at=api_key.last_used_at,
                revoked_at=api_key.revoked_at,
            )
        )
        await self._session.flush()

    async def update(self, api_key: ApiKey) -> None:
        entity = await self._session.get(ApiKeyEntity, api_key.id)
        if entity is None:
            raise NotFoundError(f"API key {api_key.id} not found")
        entity.name = api_key.name
        entity.last_used_at = api_key.last_used_at
        entity.revoked_at = api_key.revoked_at
        await self._session.flush()

    async def get(self, api_key_id: UUID) -> ApiKey | None:
        entity = await self._session.get(ApiKeyEntity, api_key_id)
        return _to_domain(entity) if entity else None

    async def get_by_hash(self, key_hash: str) -> ApiKey | None:
        entity = await self._session.scalar(
            select(ApiKeyEntity).where(ApiKeyEntity.key_hash == key_hash)
        )
        return _to_domain(entity) if entity else None

    async def list(self) -> list[ApiKey]:
        entities = await self._session.scalars(
            select(ApiKeyEntity).order_by(ApiKeyEntity.created_at)
        )
        return [_to_domain(e) for e in entities]


def _to_domain(entity: ApiKeyEntity) -> ApiKey:
    return ApiKey(
        id=entity.id,
        name=entity.name,
        display_prefix=entity.display_prefix,
        key_hash=entity.key_hash,
        created_at=entity.created_at,
        last_used_at=entity.last_used_at,
        revoked_at=entity.revoked_at,
    )
