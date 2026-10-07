"""API key use cases: issue, authenticate, list, revoke."""

from dataclasses import dataclass
from uuid import UUID

from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.domain.exceptions import AuthenticationError, NotFoundError
from app.domain.models import ApiKey
from app.domain.models.api_key import hash_api_key


@dataclass(frozen=True, slots=True)
class IssuedApiKey:
    api_key: ApiKey
    raw_key: str  # shown to the caller exactly once


class IssueApiKey:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, name: str) -> IssuedApiKey:
        api_key, raw_key = ApiKey.issue(name)
        async with self._uow_factory() as uow:
            await uow.api_keys.add(api_key)
            await uow.commit()
        return IssuedApiKey(api_key=api_key, raw_key=raw_key)


class AuthenticateApiKey:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, raw_key: str) -> ApiKey:
        async with self._uow_factory() as uow:
            api_key = await uow.api_keys.get_by_hash(hash_api_key(raw_key))
            if api_key is None or not api_key.is_active:
                # Same error either way: don't reveal whether a key exists.
                raise AuthenticationError("Invalid or revoked API key")
            if api_key.record_use():
                await uow.api_keys.update(api_key)
                await uow.commit()
        return api_key


class ListApiKeys:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self) -> list[ApiKey]:
        async with self._uow_factory() as uow:
            return await uow.api_keys.list()


class RevokeApiKey:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, api_key_id: UUID) -> ApiKey:
        async with self._uow_factory() as uow:
            api_key = await uow.api_keys.get(api_key_id)
            if api_key is None:
                raise NotFoundError(f"API key {api_key_id} not found")
            api_key.revoke()
            await uow.api_keys.update(api_key)
            await uow.commit()
        return api_key


__all__ = ["AuthenticateApiKey", "IssueApiKey", "IssuedApiKey", "ListApiKeys", "RevokeApiKey"]
