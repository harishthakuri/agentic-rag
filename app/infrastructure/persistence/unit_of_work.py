"""SQLAlchemy implementation of the Unit of Work port."""

from types import TracebackType
from typing import Self

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.application.ports.unit_of_work import UnitOfWork
from app.infrastructure.persistence.repositories.api_key_repository import (
    SqlAlchemyApiKeyRepository,
)
from app.infrastructure.persistence.repositories.chunk_repository import SqlAlchemyChunkRepository
from app.infrastructure.persistence.repositories.collection_repository import (
    SqlAlchemyCollectionRepository,
)
from app.infrastructure.persistence.repositories.document_repository import (
    SqlAlchemyDocumentRepository,
)
from app.infrastructure.persistence.repositories.ingestion_job_repository import (
    SqlAlchemyIngestionJobRepository,
)


class SqlAlchemyUnitOfWork(UnitOfWork):
    """One session and one transaction per unit of work.

    Repositories are created on `__aenter__`, sharing the same session, so all
    their changes commit (or roll back) together.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> Self:
        self._session = self._session_factory()
        self.collections = SqlAlchemyCollectionRepository(self._session)
        self.documents = SqlAlchemyDocumentRepository(self._session)
        self.chunks = SqlAlchemyChunkRepository(self._session)
        self.ingestion_jobs = SqlAlchemyIngestionJobRepository(self._session)
        self.api_keys = SqlAlchemyApiKeyRepository(self._session)
        return await super().__aenter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            await super().__aexit__(exc_type, exc, tb)
        finally:
            await self.session.close()
            self._session = None

    @property
    def session(self) -> AsyncSession:
        if self._session is None:
            raise RuntimeError("UnitOfWork used outside of 'async with'")
        return self._session

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()
