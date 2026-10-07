from uuid import UUID

from fastapi import APIRouter, status

from app.presentation.api.dependencies import DeleteDocumentDep, GetDocumentDep
from app.presentation.api.errors import PROBLEM_RESPONSES
from app.presentation.api.schemas.documents import DocumentResponse

router = APIRouter(prefix="/documents", tags=["documents"], responses=PROBLEM_RESPONSES)


@router.get("/{document_id}")
async def get_document(document_id: UUID, use_case: GetDocumentDep) -> DocumentResponse:
    """A document and its ingestion status."""
    return DocumentResponse.from_domain(await use_case.execute(document_id))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: UUID, use_case: DeleteDocumentDep) -> None:
    """Delete a document together with its chunks."""
    await use_case.execute(document_id)
