"""Unit of Work port.

A use case opens one Unit of Work, uses the repositories it exposes, and
commits once. Either everything succeeds or nothing is written. Repository
attributes are added here as the domain grows (collections, documents, ...).
"""

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self


class UnitOfWork(ABC):
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
