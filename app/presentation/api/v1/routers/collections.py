from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

import structlog
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
from fastapi.responses import StreamingResponse

from app.application.dto.pagination import PageRequest
from app.application.use_cases.answering import (
    AnswerCompleted,
    AnswerDelta,
    AskEvent,
    AskQuery,
    SourcesFound,
)
from app.application.use_cases.collections import CreateCollectionCommand
from app.application.use_cases.ingestion import UploadDocumentCommand
from app.application.use_cases.retrieval import SearchQuery
from app.presentation.api.dependencies import (
    AskQuestionDep,
    ContainerDep,
    CreateCollectionDep,
    DeleteCollectionDep,
    GetCollectionDep,
    ListCollectionsDep,
    ListDocumentsDep,
    SearchCollectionDep,
    UploadDocumentDep,
)
from app.presentation.api.errors import PROBLEM_RESPONSES
from app.presentation.api.schemas.ask import (
    AskRequest,
    AskResponse,
    DoneEvent,
    SourceResponse,
    SourcesEvent,
    TokenEvent,
)
from app.presentation.api.schemas.collections import CollectionResponse, CreateCollectionRequest
from app.presentation.api.schemas.common import Page, page_params
from app.presentation.api.schemas.documents import DocumentResponse
from app.presentation.api.schemas.jobs import IngestionJobResponse, UploadDocumentResponse
from app.presentation.api.schemas.search import SearchRequest, SearchResponse
from app.presentation.api.sse import sse_event, sse_response

logger = structlog.get_logger(__name__)

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


@router.post("/{collection_id}/search")
async def search(
    collection_id: UUID, body: SearchRequest, use_case: SearchCollectionDep
) -> SearchResponse:
    """Retrieve the chunks most relevant to a query (no LLM involved).

    Compare the modes on the same query: each hit shows its rank in the `vector`
    and `keyword` result lists, so you can see why hybrid search ranks it where it does.
    """
    result = await use_case.execute(
        SearchQuery(
            collection_id=collection_id,
            text=body.query,
            mode=body.mode,
            top_k=body.top_k,
            candidates=body.candidates,
            rerank=body.rerank,
        )
    )
    return SearchResponse.from_domain(result)


@router.post(
    "/{collection_id}/ask",
    response_model=AskResponse,
    responses={
        200: {
            "description": "JSON answer, or with `stream: true` a `text/event-stream` of "
            "`sources`, `token` (repeated), then `done` (or `error`) events",
            "content": {"text/event-stream": {}},
        },
        503: PROBLEM_RESPONSES[422],
    },
)
async def ask(
    collection_id: UUID, body: AskRequest, use_case: AskQuestionDep
) -> AskResponse | StreamingResponse:
    """Answer a question from the collection's documents, citing sources as [n].

    Retrieval (hybrid search + rerank) runs once with the question; the chat model
    then answers using only the retrieved sources. Every response lists the sources
    the model was given and which of them it cited.
    """
    query = AskQuery(
        collection_id=collection_id,
        question=body.question,
        mode=body.mode,
        top_k=body.top_k,
        rerank=body.rerank,
    )
    if not body.stream:
        return AskResponse.from_domain(await use_case.execute(query))

    events = aiter(use_case.stream(query))
    # Run retrieval before sending headers: errors like 404 (unknown collection) or
    # 503 (model down) still become proper HTTP responses, not a broken stream.
    first = await anext(events)
    return sse_response(_sse(first, events))


async def _sse(first: AskEvent, rest: AsyncIterator[AskEvent]) -> AsyncIterator[str]:
    try:
        yield _to_sse(first)
        async for event in rest:
            yield _to_sse(event)
    except Exception as exc:  # headers are sent: report in-band, then end the stream
        logger.exception("ask.stream_failed")
        yield sse_event(
            "error", {"title": "Answer generation failed", "detail": type(exc).__name__}
        )


def _to_sse(event: AskEvent) -> str:
    match event:
        case SourcesFound(sources=sources, search=search):
            payload = SourcesEvent(
                sources=[SourceResponse.from_domain(s) for s in sources],
                reranker=search.reranker,
            )
            return sse_event("sources", payload)
        case AnswerDelta(text=text):
            return sse_event("token", TokenEvent(text=text))
        case AnswerCompleted():
            return sse_event("done", DoneEvent.from_domain(event))
