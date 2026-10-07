"""FastAPI dependencies: resolve use cases from the composition root."""

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.application.use_cases.collections import (
    CreateCollection,
    DeleteCollection,
    GetCollection,
    ListCollections,
)
from app.application.use_cases.documents import DeleteDocument, GetDocument, ListDocuments
from app.application.use_cases.ingestion import GetIngestionJob, UploadDocument
from app.application.use_cases.retrieval import SearchCollection
from app.application.use_cases.system.check_readiness import CheckReadiness
from app.bootstrap.container import Container
from app.domain.exceptions import AuthenticationError
from app.domain.models import ApiKey


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]

# --- Authentication -----------------------------------------------------------
_bearer = HTTPBearer(auto_error=False, description="API key: `Authorization: Bearer srag_...`")


async def require_api_key(
    container: ContainerDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> ApiKey | None:
    """Authenticate the caller. Returns None only when auth is disabled in settings."""
    if not container.settings.auth_enabled:
        return None
    if credentials is None:
        raise AuthenticationError("Missing API key (Authorization: Bearer <key>)")
    return await container.authenticate_api_key().execute(credentials.credentials)


# --- Use cases ----------------------------------------------------------------
def _use_case[T](factory: Callable[[Container], T]) -> Callable[[Container], T]:
    """Turn a container factory call into a FastAPI dependency.

    The factory is looked up on each request (not bound at import), so tests can
    substitute a container method.
    """

    def dependency(container: ContainerDep) -> T:
        return factory(container)

    return dependency


CheckReadinessDep = Annotated[CheckReadiness, Depends(_use_case(lambda c: c.check_readiness()))]
CreateCollectionDep = Annotated[
    CreateCollection, Depends(_use_case(lambda c: c.create_collection()))
]
GetCollectionDep = Annotated[GetCollection, Depends(_use_case(lambda c: c.get_collection()))]
ListCollectionsDep = Annotated[ListCollections, Depends(_use_case(lambda c: c.list_collections()))]
DeleteCollectionDep = Annotated[
    DeleteCollection, Depends(_use_case(lambda c: c.delete_collection()))
]
ListDocumentsDep = Annotated[ListDocuments, Depends(_use_case(lambda c: c.list_documents()))]
GetDocumentDep = Annotated[GetDocument, Depends(_use_case(lambda c: c.get_document()))]
DeleteDocumentDep = Annotated[DeleteDocument, Depends(_use_case(lambda c: c.delete_document()))]
UploadDocumentDep = Annotated[UploadDocument, Depends(_use_case(lambda c: c.upload_document()))]
GetIngestionJobDep = Annotated[GetIngestionJob, Depends(_use_case(lambda c: c.get_ingestion_job()))]
SearchCollectionDep = Annotated[
    SearchCollection, Depends(_use_case(lambda c: c.search_collection()))
]
