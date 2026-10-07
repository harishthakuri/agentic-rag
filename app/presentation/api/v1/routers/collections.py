from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)

from app.application.dto.pagination import PageRequest
from app.application.use_cases.collections import CreateCollectionCommand
from app.application.use_cases.ingestion import UploadDocumentCommand
from app.presentation.api.dependencies import (
    ContainerDep,
    CreateCollectionDep,
    DeleteCollectionDep,
    GetCollectionDep,
    ListCollectionsDep,
    ListDocumentsDep,
    UploadDocumentDep,
)
from app.presentation.api.errors import PROBLEM_RESPONSES
from app.presentation.api.schemas.collections import CollectionResponse, CreateCollectionRequest
from app.presentation.api.schemas.common import Page, page_params
from app.presentation.api.schemas.documents import DocumentResponse
from app.presentation.api.schemas.jobs import IngestionJobResponse, UploadDocumentResponse

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


@router.post(
    "/{collection_id}/documents",
    status_code=status.HTTP_202_ACCEPTED,
    responses={status.HTTP_413_CONTENT_TOO_LARGE: PROBLEM_RESPONSES[422]},
)
async def upload_document(
    collection_id: UUID,
    use_case: UploadDocumentDep,
    container: ContainerDep,
    request: Request,
    response: Response,
    file: Annotated[UploadFile, File(description="Markdown (.md), text (.txt) or PDF (.pdf)")],
    title: Annotated[str | None, Form(max_length=500)] = None,
) -> UploadDocumentResponse:
    """Upload a document for ingestion.

    Returns **202 Accepted** immediately: parsing, chunking and embedding happen in
    the background worker (`make worker`). Poll the job (see the `Location` header)
    or the document until its status is `ready`.
    """
    max_bytes = container.settings.max_upload_mb * 1024 * 1024
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"The file exceeds the {container.settings.max_upload_mb} MB upload limit",
        )
    uploaded = await use_case.execute(
        UploadDocumentCommand(
            collection_id=collection_id,
            filename=file.filename or "upload",
            content=content,
            title=title,
        )
    )
    response.headers["Location"] = str(request.url_for("get_job", job_id=uploaded.job.id))
    return UploadDocumentResponse(
        document=DocumentResponse.from_domain(uploaded.document),
        job=IngestionJobResponse.from_domain(uploaded.job),
    )
