import pytest

from app.application.dto.pagination import PageRequest
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.use_cases.api_keys import (
    AuthenticateApiKey,
    IssueApiKey,
    RevokeApiKey,
)
from app.application.use_cases.collections import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    CreateCollection,
    CreateCollectionCommand,
    DeleteCollection,
    GetCollection,
    ListCollections,
)
from app.application.use_cases.documents import ListDocuments
from app.domain.exceptions import AuthenticationError, DomainValidationError
from app.domain.value_objects import EmbeddingSpec, new_id
from tests.fakes import InMemoryStore, InMemoryUnitOfWork

EMBEDDING = EmbeddingSpec(model="qwen3-embedding:8b", dimensions=1024)


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def uow_factory(store: InMemoryStore) -> UnitOfWorkFactory:
    return lambda: InMemoryUnitOfWork(store)


# --- Collections --------------------------------------------------------------
async def test_create_collection_binds_configured_embedding_model(
    uow_factory: UnitOfWorkFactory, store: InMemoryStore
) -> None:
    collection = await CreateCollection(uow_factory, EMBEDDING).execute(
        CreateCollectionCommand(name="  K8s-Docs ", description="Kubernetes")
    )

    assert collection.name.value == "k8s-docs"
    assert collection.embedding == EMBEDDING
    assert store.collections[collection.id] == collection
    assert store.commits == 1


async def test_create_collection_rejects_duplicate_name(uow_factory: UnitOfWorkFactory) -> None:
    create = CreateCollection(uow_factory, EMBEDDING)
    await create.execute(CreateCollectionCommand(name="docs"))

    with pytest.raises(CollectionAlreadyExistsError):
        await create.execute(CreateCollectionCommand(name="DOCS"))


async def test_create_collection_rejects_invalid_name_before_touching_storage(
    uow_factory: UnitOfWorkFactory, store: InMemoryStore
) -> None:
    with pytest.raises(DomainValidationError):
        await CreateCollection(uow_factory, EMBEDDING).execute(
            CreateCollectionCommand(name="no spaces allowed")
        )
    assert store.commits == 0


async def test_get_and_delete_unknown_collection_raise_not_found(
    uow_factory: UnitOfWorkFactory,
) -> None:
    with pytest.raises(CollectionNotFoundError):
        await GetCollection(uow_factory).execute(new_id())
    with pytest.raises(CollectionNotFoundError):
        await DeleteCollection(uow_factory).execute(new_id())


async def test_list_collections_paginates_by_name(uow_factory: UnitOfWorkFactory) -> None:
    create = CreateCollection(uow_factory, EMBEDDING)
    for name in ("charlie", "alpha", "bravo"):
        await create.execute(CreateCollectionCommand(name=name))

    page = await ListCollections(uow_factory).execute(PageRequest(limit=2, offset=1))

    assert [c.name.value for c in page] == ["bravo", "charlie"]


async def test_list_documents_of_unknown_collection_raises(uow_factory: UnitOfWorkFactory) -> None:
    with pytest.raises(CollectionNotFoundError):
        await ListDocuments(uow_factory).execute(new_id(), PageRequest())


def test_page_request_bounds() -> None:
    with pytest.raises(DomainValidationError):
        PageRequest(limit=0)
    with pytest.raises(DomainValidationError):
        PageRequest(limit=101)
    with pytest.raises(DomainValidationError):
        PageRequest(offset=-1)


# --- API keys -----------------------------------------------------------------
async def test_issued_key_authenticates(uow_factory: UnitOfWorkFactory) -> None:
    issued = await IssueApiKey(uow_factory).execute("dev")

    api_key = await AuthenticateApiKey(uow_factory).execute(issued.raw_key)

    assert api_key.id == issued.api_key.id
    assert api_key.last_used_at is not None


async def test_unknown_and_revoked_keys_are_rejected_identically(
    uow_factory: UnitOfWorkFactory,
) -> None:
    issued = await IssueApiKey(uow_factory).execute("dev")
    await RevokeApiKey(uow_factory).execute(issued.api_key.id)
    authenticate = AuthenticateApiKey(uow_factory)

    with pytest.raises(AuthenticationError) as revoked:
        await authenticate.execute(issued.raw_key)
    with pytest.raises(AuthenticationError) as unknown:
        await authenticate.execute("srag_does-not-exist")

    assert revoked.value.message == unknown.value.message
