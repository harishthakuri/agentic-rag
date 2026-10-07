"""Version 1 API router. Every route below requires an API key (unless auth is disabled)."""

from fastapi import APIRouter, Depends

from app.presentation.api.dependencies import require_api_key
from app.presentation.api.v1.routers import collections, documents

api_v1_router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])
api_v1_router.include_router(collections.router)
api_v1_router.include_router(documents.router)
