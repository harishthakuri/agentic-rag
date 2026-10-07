"""Error responses as RFC 9457 Problem Details (`application/problem+json`).

Domain and application errors are mapped to HTTP status codes in exactly one
place, so use cases never know about HTTP.
"""

from collections.abc import Mapping
from http import HTTPStatus
from typing import Any, cast

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.domain.exceptions import (
    AuthenticationError,
    ConflictError,
    DomainError,
    DomainValidationError,
    NotFoundError,
)

PROBLEM_JSON = "application/problem+json"

logger = structlog.get_logger(__name__)


class ProblemDetails(BaseModel):
    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None


# Most specific first: the first matching class wins.
_DOMAIN_STATUS: list[tuple[type[DomainError], HTTPStatus]] = [
    (AuthenticationError, HTTPStatus.UNAUTHORIZED),
    (NotFoundError, HTTPStatus.NOT_FOUND),
    (ConflictError, HTTPStatus.CONFLICT),
    (DomainValidationError, HTTPStatus.UNPROCESSABLE_ENTITY),
]

# For OpenAPI: `responses=PROBLEM_RESPONSES` documents the error shape on a route.
PROBLEM_RESPONSES: dict[int | str, dict[str, Any]] = {
    status: {"model": ProblemDetails, "content": {PROBLEM_JSON: {}}}
    for status in (401, 404, 409, 422)
}


def problem_response(
    request: Request,
    status: HTTPStatus,
    detail: str | None = None,
    headers: Mapping[str, str] | None = None,
    **extensions: Any,
) -> JSONResponse:
    body = ProblemDetails(
        title=status.phrase, status=status.value, detail=detail, instance=request.url.path
    ).model_dump(exclude_none=True)
    return JSONResponse(
        {**body, **extensions}, status_code=status.value, media_type=PROBLEM_JSON, headers=headers
    )


# Starlette types every handler as (Request, Exception); each is registered for one type.
async def _domain_error(request: Request, exc: Exception) -> JSONResponse:
    error = cast("DomainError", exc)
    status = next(
        (code for cls, code in _DOMAIN_STATUS if isinstance(error, cls)), HTTPStatus.BAD_REQUEST
    )
    headers = {"WWW-Authenticate": "Bearer"} if status is HTTPStatus.UNAUTHORIZED else None
    return problem_response(request, status, error.message, headers)


async def _request_validation_error(request: Request, exc: Exception) -> JSONResponse:
    errors = [
        {"location": list(err["loc"]), "message": err["msg"], "type": err["type"]}
        for err in cast("RequestValidationError", exc).errors()
    ]
    return problem_response(
        request, HTTPStatus.UNPROCESSABLE_ENTITY, "Request validation failed", errors=errors
    )


async def _http_exception(request: Request, exc: Exception) -> JSONResponse:
    error = cast("StarletteHTTPException", exc)
    return problem_response(request, HTTPStatus(error.status_code), error.detail, error.headers)


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    # Details stay in the logs; clients get a generic message.
    logger.error("request.unhandled_error", error_type=type(exc).__name__)
    return problem_response(request, HTTPStatus.INTERNAL_SERVER_ERROR, "Unexpected error")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(DomainError, _domain_error)
    app.add_exception_handler(RequestValidationError, _request_validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_exception)
    app.add_exception_handler(Exception, _unhandled)
