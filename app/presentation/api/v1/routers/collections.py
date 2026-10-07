from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response, status

from app.application.dto.pagination import PageRequest
from app.application.use_cases.collections import CreateCollectionCommand
from app.presentation.api.dependencies import (
    CreateCollectionDep,
    DeleteCollectionDep,
    GetCollectionDep,
    ListCollectionsDep,
    ListDocumentsDep,
)
from app.presentation.api.errors import PROBLEM_RESPONSES
from app.presentation.api.schemas.collections import CollectionResponse, CreateCollectionRequest
from app.presentation.api.schemas.common import Page, page_params
from app.presentation.api.schemas.documents import DocumentResponse

router = APIRouter(prefix="/collections", tags=["collections"], responses=PROBLEM_RESPONSES)

PageDep = Annotated[PageRequest, Depends(page_params)]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_collection(
    body: CreateCollectionRequest,
    use_case: CreateCollectionDep,
    request: Request,
    response: Response,
) -> CollectionResponse:
    """Create a collection. It is bound to the currently configured embedding model."""
    collection = await use_case.execute(
        CreateCollectionCommand(name=body.name, description=body.description)
    )
    response.headers["Location"] = str(
        request.url_for("get_collection", collection_id=collection.id)
    )
    return CollectionResponse.from_domain(collection)


@router.get("")
async def list_collections(use_case: ListCollectionsDep, page: PageDep) -> Page[CollectionResponse]:
    collections = await use_case.execute(page)
    return Page(
        items=[CollectionResponse.from_domain(c) for c in collections],
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/{collection_id}")
async def get_collection(collection_id: UUID, use_case: GetCollectionDep) -> CollectionResponse:
    return CollectionResponse.from_domain(await use_case.execute(collection_id))


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(collection_id: UUID, use_case: DeleteCollectionDep) -> None:
    """Delete a collection together with all of its documents and chunks."""
    await use_case.execute(collection_id)


@router.get("/{collection_id}/documents")
async def list_documents(
    collection_id: UUID, use_case: ListDocumentsDep, page: PageDep
) -> Page[DocumentResponse]:
    documents = await use_case.execute(collection_id, page)
    return Page(
        items=[DocumentResponse.from_domain(d) for d in documents],
        limit=page.limit,
        offset=page.offset,
    )
