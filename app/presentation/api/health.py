"""Liveness and readiness probes (unversioned, unauthenticated)."""

from fastapi import APIRouter, Response, status

from app.presentation.api.dependencies import CheckReadinessDep
from app.presentation.api.schemas.health import (
    DependencyStatusResponse,
    LivenessResponse,
    ReadinessResponse,
)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=LivenessResponse)
async def live() -> LivenessResponse:
    """The process is up and serving requests."""
    return LivenessResponse()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
async def ready(check_readiness: CheckReadinessDep, response: Response) -> ReadinessResponse:
    """All dependencies (database, later: LLM and embedding endpoints) are reachable."""
    report = await check_readiness.execute()
    if not report.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if report.ready else "not_ready",
        dependencies=[
            DependencyStatusResponse(
                name=dep.name, healthy=dep.healthy, latency_ms=dep.latency_ms, error=dep.error
            )
            for dep in report.dependencies
        ],
    )
