"""Composition root.

This is the only module that knows about every concrete adapter. It wires
infrastructure implementations to application ports and hands fully built use
cases to the presentation layer. Swapping an adapter (e.g. Ollama → OpenAI,
LLM reranker → cross-encoder) is a change here and in settings, nowhere else.
"""

from app.application.ports.unit_of_work import UnitOfWork
from app.application.use_cases.api_keys import (
    AuthenticateApiKey,
    IssueApiKey,
    ListApiKeys,
    RevokeApiKey,
)
from app.application.use_cases.collections import (
    CreateCollection,
    DeleteCollection,
    GetCollection,
    ListCollections,
)
from app.application.use_cases.documents import DeleteDocument, GetDocument, ListDocuments
from app.application.use_cases.system.check_readiness import CheckReadiness
from app.core.config import Settings
from app.domain.value_objects import EmbeddingSpec
from app.infrastructure.persistence.database import Database
from app.infrastructure.persistence.health import DatabaseHealthCheck
from app.infrastructure.persistence.orm import EMBEDDING_DIMENSIONS
from app.infrastructure.persistence.unit_of_work import SqlAlchemyUnitOfWork


class ConfigurationError(RuntimeError):
    pass


class Container:
    def __init__(self, settings: Settings) -> None:
        if settings.embedding_dim != EMBEDDING_DIMENSIONS:
            raise ConfigurationError(
                f"EMBEDDING_DIM={settings.embedding_dim} does not match the database column "
                f"vector({EMBEDDING_DIMENSIONS}). Changing it requires a migration and "
                "re-embedding every collection."
            )
        self.settings = settings
        self.database = Database(settings)
        self.embedding_spec = EmbeddingSpec(
            model=settings.embedding_model, dimensions=settings.embedding_dim
        )

    # --- Infrastructure ----------------------------------------------------
    def unit_of_work(self) -> UnitOfWork:
        return SqlAlchemyUnitOfWork(self.database.session_factory)

    # --- System ------------------------------------------------------------
    def check_readiness(self) -> CheckReadiness:
        return CheckReadiness(checks=[DatabaseHealthCheck(self.database)])

    # --- Collections -------------------------------------------------------
    def create_collection(self) -> CreateCollection:
        return CreateCollection(self.unit_of_work, self.embedding_spec)

    def get_collection(self) -> GetCollection:
        return GetCollection(self.unit_of_work)

    def list_collections(self) -> ListCollections:
        return ListCollections(self.unit_of_work)

    def delete_collection(self) -> DeleteCollection:
        return DeleteCollection(self.unit_of_work)

    # --- Documents ---------------------------------------------------------
    def list_documents(self) -> ListDocuments:
        return ListDocuments(self.unit_of_work)

    def get_document(self) -> GetDocument:
        return GetDocument(self.unit_of_work)

    def delete_document(self) -> DeleteDocument:
        return DeleteDocument(self.unit_of_work)

    # --- API keys ----------------------------------------------------------
    def authenticate_api_key(self) -> AuthenticateApiKey:
        return AuthenticateApiKey(self.unit_of_work)

    def issue_api_key(self) -> IssueApiKey:
        return IssueApiKey(self.unit_of_work)

    def list_api_keys(self) -> ListApiKeys:
        return ListApiKeys(self.unit_of_work)

    def revoke_api_key(self) -> RevokeApiKey:
        return RevokeApiKey(self.unit_of_work)

    # --- Lifecycle ---------------------------------------------------------
    async def aclose(self) -> None:
        await self.database.dispose()
