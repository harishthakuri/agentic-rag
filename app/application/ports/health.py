"""Health check port: one implementation per external dependency."""

from typing import Protocol


class HealthCheck(Protocol):
    @property
    def name(self) -> str: ...

    async def check(self) -> None:
        """Return normally if healthy; raise any exception if not."""
        ...
