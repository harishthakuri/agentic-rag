from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.domain.models import IngestionJob, JobStatus
from app.presentation.api.schemas.documents import DocumentResponse


class IngestionJobResponse(BaseModel):
    id: UUID
    document_id: UUID
    status: JobStatus
    attempts: int
    max_attempts: int
    last_error: str | None
    run_after: datetime
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, job: IngestionJob) -> "IngestionJobResponse":
        return cls(
            id=job.id,
            document_id=job.document_id,
            status=job.status,
            attempts=job.attempts,
            max_attempts=job.max_attempts,
            last_error=job.last_error,
            run_after=job.run_after,
            finished_at=job.finished_at,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )


class UploadDocumentResponse(BaseModel):
    document: DocumentResponse
    job: IngestionJobResponse
