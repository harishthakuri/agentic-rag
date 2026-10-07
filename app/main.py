"""FastAPI application factory.

Run with:  uv run uvicorn app.main:create_app --factory --reload
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.bootstrap.container import Container
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.presentation.api import health
from app.presentation.api.errors import register_error_handlers
from app.presentation.api.middleware import request_context_middleware
from app.presentation.api.v1.router import api_v1_router
from app.presentation.web import mount_ui

logger = structlog.get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = Container(settings)
        app.state.container = container
        logger.info("app.started", env=settings.app_env, llm_model=settings.llm_model)
        try:
            yield
        finally:
            await container.aclose()
            logger.info("app.stopped")

    app = FastAPI(
        title=settings.app_name,
        description="Retrieval-Augmented Generation with hybrid and agentic search.",
        version="0.1.0",
        lifespan=lifespan,
    )

    register_error_handlers(app)
    app.middleware("http")(request_context_middleware)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    app.include_router(health.router)
    app.include_router(api_v1_router)
    mount_ui(app)
    return app
