from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.exceptions import NotFoundError
from app.domain.models import IngestionJob, JobStatus
from app.domain.repositories import IngestionJobRepository
from app.infrastructure.persistence.orm import IngestionJobEntity as Job


class SqlAlchemyIngestionJobRepository(IngestionJobRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, job: IngestionJob) -> None:
        entity = Job(id=job.id, created_at=job.created_at)
        _apply(job, entity)
        self._session.add(entity)
        await self._session.flush()

    async def update(self, job: IngestionJob) -> None:
        entity = await self._session.get(Job, job.id)
        if entity is None:
            raise NotFoundError(f"Ingestion job {job.id} not found")
        _apply(job, entity)
        await self._session.flush()

    async def get(self, job_id: UUID) -> IngestionJob | None:
        entity = await self._session.get(Job, job_id)
        return _to_domain(entity) if entity else None

    async def claim_next(self, worker_id: str, lease: timedelta) -> IngestionJob | None:
        """
        UPDATE ingestion_jobs SET status = 'running', attempts = attempts + 1, ...
        WHERE id = (
            SELECT id FROM ingestion_jobs
            WHERE (status = 'queued' AND run_after <= :now)
               OR (status = 'running' AND locked_at < :now - :lease)   -- crashed worker
            ORDER BY run_after
            LIMIT 1
            FOR UPDATE SKIP LOCKED      -- rows locked by other workers are skipped, not waited on
        )
        RETURNING *

        `:now` is the application's clock, not the database's now(): the domain sets
        run_after (backoff) and locked_at with the application clock, and comparing
        those against a different clock breaks as soon as the two drift apart.
        """
        now = datetime.now(UTC)
        due = or_(
            and_(Job.status == JobStatus.QUEUED.value, Job.run_after <= now),
            and_(
                Job.status == JobStatus.RUNNING.value,
                Job.locked_at < now - lease,
                Job.attempts < Job.max_attempts,
            ),
        )
        candidate = (
            select(Job.id)
            .where(due)
            .order_by(Job.run_after)
            .limit(1)
            .with_for_update(skip_locked=True)
            .scalar_subquery()
        )
        statement = (
            update(Job)
            .where(Job.id == candidate)
            .values(
                status=JobStatus.RUNNING.value,
                attempts=Job.attempts + 1,
                locked_at=now,
                locked_by=worker_id,
                updated_at=now,
            )
            .returning(Job)
            .execution_options(synchronize_session=False)
        )
        entity = (await self._session.execute(statement)).scalar_one_or_none()
        return _to_domain(entity) if entity else None


def _to_domain(entity: Job) -> IngestionJob:
    return IngestionJob(
        id=entity.id,
        document_id=entity.document_id,
        status=JobStatus(entity.status),
        attempts=entity.attempts,
        max_attempts=entity.max_attempts,
        last_error=entity.last_error,
        run_after=entity.run_after,
        locked_at=entity.locked_at,
        locked_by=entity.locked_by,
        finished_at=entity.finished_at,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


def _apply(job: IngestionJob, entity: Job) -> None:
    entity.document_id = job.document_id
    entity.status = job.status.value
    entity.attempts = job.attempts
    entity.max_attempts = job.max_attempts
    entity.last_error = job.last_error
    entity.run_after = job.run_after
    entity.locked_at = job.locked_at
    entity.locked_by = job.locked_by
    entity.finished_at = job.finished_at
    entity.updated_at = job.updated_at
