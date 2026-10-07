"""Composition root.

This is the only module that knows about every concrete adapter. It wires
infrastructure implementations to application ports and hands fully built use
cases to the presentation layer. Swapping an adapter (e.g. Ollama → OpenAI,
LLM reranker → cross-encoder) is a change here and in settings, nowhere else.
"""

from app.application.ports.unit_of_work import UnitOfWork
from app.application.use_cases.system.check_readiness import CheckReadiness
from app.core.config import Settings
from app.infrastructure.persistence.database import Database
from app.infrastructure.persistence.health import DatabaseHealthCheck
from app.infrastructure.persistence.unit_of_work import SqlAlchemyUnitOfWork


class Container:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = Database(settings)

    # --- Factories (a fresh instance per request / job) --------------------
    def unit_of_work(self) -> UnitOfWork:
        return SqlAlchemyUnitOfWork(self.database.session_factory)

    def check_readiness(self) -> CheckReadiness:
        return CheckReadiness(checks=[DatabaseHealthCheck(self.database)])

    # --- Lifecycle ---------------------------------------------------------
    async def aclose(self) -> None:
        await self.database.dispose()
