"""Readiness use case: are all dependencies this service needs reachable?"""

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass

from app.application.ports.health import HealthCheck


@dataclass(frozen=True, slots=True)
class DependencyStatus:
    name: str
    healthy: bool
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    dependencies: tuple[DependencyStatus, ...]

    @property
    def ready(self) -> bool:
        return all(dep.healthy for dep in self.dependencies)


class CheckReadiness:
    """Runs every health check concurrently, each bounded by a timeout."""

    def __init__(self, checks: Sequence[HealthCheck], timeout_seconds: float = 3.0) -> None:
        self._checks = checks
        self._timeout = timeout_seconds

    async def execute(self) -> ReadinessReport:
        results = await asyncio.gather(*(self._run(check) for check in self._checks))
        return ReadinessReport(dependencies=tuple(results))

    async def _run(self, check: HealthCheck) -> DependencyStatus:
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout):
                await check.check()
        except Exception as exc:
            error = "timed out" if isinstance(exc, TimeoutError) else type(exc).__name__
            return DependencyStatus(check.name, healthy=False, latency_ms=_ms(started), error=error)
        return DependencyStatus(check.name, healthy=True, latency_ms=_ms(started))


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
