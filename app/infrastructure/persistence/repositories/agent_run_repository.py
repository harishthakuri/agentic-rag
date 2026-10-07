from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.exceptions import NotFoundError
from app.domain.models import AgentRun, AgentStep, RunStatus
from app.domain.repositories import AgentRunRepository
from app.infrastructure.persistence.orm import AgentRunEntity, AgentStepEntity


class SqlAlchemyAgentRunRepository(AgentRunRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, run: AgentRun) -> None:
        entity = AgentRunEntity(id=run.id, created_at=run.created_at)
        _apply(run, entity)
        self._session.add(entity)
        await self._session.flush()

    async def update(self, run: AgentRun) -> None:
        entity = await self._session.get(AgentRunEntity, run.id)
        if entity is None:
            raise NotFoundError(f"Agent run {run.id} not found")
        _apply(run, entity)
        await self._session.flush()

    async def get(self, run_id: UUID) -> AgentRun | None:
        entity = await self._session.get(AgentRunEntity, run_id)
        return _to_domain(entity) if entity else None

    async def add_step(self, step: AgentStep) -> None:
        self._session.add(
            AgentStepEntity(
                id=step.id,
                run_id=step.run_id,
                number=step.number,
                tool=step.tool,
                arguments=step.arguments,
                result=step.result,
                latency_ms=step.latency_ms,
                created_at=step.created_at,
            )
        )
        await self._session.flush()

    async def steps(self, run_id: UUID) -> list[AgentStep]:
        entities = await self._session.scalars(
            select(AgentStepEntity)
            .where(AgentStepEntity.run_id == run_id)
            .order_by(AgentStepEntity.number)
        )
        return [
            AgentStep(
                id=e.id,
                run_id=e.run_id,
                number=e.number,
                tool=e.tool,
                arguments=dict(e.arguments),
                result=dict(e.result),
                latency_ms=e.latency_ms,
                created_at=e.created_at,
            )
            for e in entities
        ]


def _to_domain(entity: AgentRunEntity) -> AgentRun:
    return AgentRun(
        id=entity.id,
        collection_id=entity.collection_id,
        question=entity.question,
        model=entity.model,
        status=RunStatus(entity.status),
        answer=entity.answer,
        error=entity.error,
        step_count=entity.step_count,
        prompt_tokens=entity.prompt_tokens,
        completion_tokens=entity.completion_tokens,
        latency_ms=entity.latency_ms,
        created_at=entity.created_at,
        finished_at=entity.finished_at,
    )


def _apply(run: AgentRun, entity: AgentRunEntity) -> None:
    entity.collection_id = run.collection_id
    entity.question = run.question
    entity.model = run.model
    entity.status = run.status.value
    entity.answer = run.answer
    entity.error = run.error
    entity.step_count = run.step_count
    entity.prompt_tokens = run.prompt_tokens
    entity.completion_tokens = run.completion_tokens
    entity.latency_ms = run.latency_ms
    entity.finished_at = run.finished_at
