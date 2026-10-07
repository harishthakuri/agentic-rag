from uuid import UUID

from fastapi import APIRouter

from app.presentation.api.dependencies import GetIngestionJobDep
from app.presentation.api.errors import PROBLEM_RESPONSES
from app.presentation.api.schemas.jobs import IngestionJobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"], responses=PROBLEM_RESPONSES)


@router.get("/{job_id}")
async def get_job(job_id: UUID, use_case: GetIngestionJobDep) -> IngestionJobResponse:
    """Ingestion progress: queued → running → succeeded / failed (with retries)."""
    return IngestionJobResponse.from_domain(await use_case.execute(job_id))
