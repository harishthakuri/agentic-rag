from typing import Annotated

from fastapi import Query
from pydantic import BaseModel

from app.application.dto.pagination import MAX_PAGE_SIZE, PageRequest


class Page[T](BaseModel):
    items: list[T]
    limit: int
    offset: int


def page_params(
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> PageRequest:
    return PageRequest(limit=limit, offset=offset)
