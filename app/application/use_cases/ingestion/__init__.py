"""Ingestion use cases: upload a document, process queued jobs, inspect jobs."""

from uuid import UUID

from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.use_cases.ingestion.process_job import (
    JobOutcome,
    PermanentIngestionError,
    ProcessNextIngestionJob,
)
from app.application.use_cases.ingestion.upload_document import (
    UploadDocument,
    UploadDocumentCommand,
    UploadedDocument,
)
from app.domain.exceptions import NotFoundError
from app.domain.models import IngestionJob


class IngestionJobNotFoundError(NotFoundError):
    def __init__(self, job_id: UUID) -> None:
        super().__init__(f"Ingestion job {job_id} not found")


class GetIngestionJob:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, job_id: UUID) -> IngestionJob:
        async with self._uow_factory() as uow:
            job = await uow.ingestion_jobs.get(job_id)
        if job is None:
            raise IngestionJobNotFoundError(job_id)
        return job


__all__ = [
    "GetIngestionJob",
    "IngestionJobNotFoundError",
    "JobOutcome",
    "PermanentIngestionError",
    "ProcessNextIngestionJob",
    "UploadDocument",
    "UploadDocumentCommand",
    "UploadedDocument",
]
