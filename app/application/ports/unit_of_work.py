"""Unit of Work port.

A use case opens one Unit of Work, uses the repositories it exposes, and
commits once. Either everything succeeds or nothing is written.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from types import TracebackType
from typing import Self

from app.domain.repositories import (
    ApiKeyRepository,
    ChunkRepository,
    CollectionRepository,
    DocumentRepository,
    IngestionJobRepository,
)


class UnitOfWork(ABC):
    collections: CollectionRepository
    documents: DocumentRepository
    chunks: ChunkRepository
    ingestion_jobs: IngestionJobRepository
    api_keys: ApiKeyRepository

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Anything not explicitly committed is rolled back.
        await self.rollback()

    @abstractmethod
    async def commit(self) -> None: ...

    @abstractmethod
    async def rollback(self) -> None: ...


# Use cases receive a factory and open a fresh Unit of Work per operation.
UnitOfWorkFactory = Callable[[], UnitOfWork]
