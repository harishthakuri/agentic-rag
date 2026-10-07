import pytest

from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.use_cases.collections import CollectionNotFoundError
from app.application.use_cases.retrieval import SearchCollection, SearchMode, SearchQuery
from app.domain.exceptions import ConflictError, DomainValidationError
from app.domain.models import Chunk, Collection, Document, DocumentType
from app.domain.services.rank_fusion import reciprocal_rank_fusion
from app.domain.value_objects import CollectionName, ContentHash, EmbeddingSpec, new_id
from tests.fakes import FakeEmbedder, FakeReranker, InMemoryStore, InMemoryUnitOfWork

SPEC = EmbeddingSpec(model="test-embedder", dimensions=8)


# --- Reciprocal Rank Fusion ---------------------------------------------------
def test_rrf_rewards_agreement_between_lists() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "d"]], k=60)

    # "b" is 2nd in both lists: 2/62 beats "a" (1st in one list only: 1/61).
    assert [item for item, _ in fused] == ["c", "b", "a", "d"]
    assert dict(fused)["b"] == pytest.approx(2 / 62)
    assert dict(fused)["a"] == pytest.approx(1 / 61)


def test_rrf_single_list_preserves_order_and_ties_are_deterministic() -> None:
    assert [i for i, _ in reciprocal_rank_fusion([["x", "y", "z"]])] == ["x", "y", "z"]
    assert [i for i, _ in reciprocal_rank_fusion([["p"], ["q"]])] == ["p", "q"]  # tie


def test_rrf_small_k_favours_top_ranks() -> None:
    lists = [["a", "b"], ["b", "c"], ["a", "d"]]
    # With k=0, being 1st twice ("a": 1+1) dominates; with large k, ranks matter less.
    assert reciprocal_rank_fusion(lists, k=0)[0][0] == "a"


# --- SearchCollection ---------------------------------------------------------
@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def uow(store: InMemoryStore) -> UnitOfWorkFactory:
    return lambda: InMemoryUnitOfWork(store)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(SPEC)


@pytest.fixture
async def collection(store: InMemoryStore, embedder: FakeEmbedder) -> Collection:
    return await seed_collection(store, embedder)


async def seed_collection(store: InMemoryStore, embedder: FakeEmbedder) -> Collection:
    """Three chunks. The fake embedder maps identical texts to identical vectors,
    so a query equal to a chunk's contextual text is that chunk's nearest neighbour."""
    collection = Collection.create(name=CollectionName("docs"), embedding=SPEC)
    document = Document(
        collection_id=collection.id,
        title="Guide",
        source_filename="guide.md",
        document_type=DocumentType.MARKDOWN,
        content_hash=ContentHash.of(b"guide"),
        size_bytes=5,
        storage_key="k",
    )
    texts = ["clusterip services expose pods", "ingress routes http", "etag validates caches"]
    chunks = [
        Chunk(
            document_id=document.id,
            collection_id=collection.id,
            ordinal=i,
            text=t,
            contextual_text=t,
            token_count=len(t.split()),
            embedding=(await embedder.embed_documents([t]))[0],
        )
        for i, t in enumerate(texts)
    ]
    store.collections[collection.id] = collection
    store.documents[document.id] = document
    store.chunks[document.id] = chunks
    return collection


async def test_vector_mode_finds_nearest_and_reports_vector_rank(
    uow: UnitOfWorkFactory, embedder: FakeEmbedder, collection: Collection
) -> None:
    result = await SearchCollection(uow, embedder).execute(
        SearchQuery(collection.id, "ingress routes http", mode=SearchMode.VECTOR, top_k=2)
    )

    top = result.hits[0]
    assert top.match.text == "ingress routes http"
    assert top.vector_rank == 1 and top.vector_similarity == pytest.approx(1.0)
    assert top.keyword_rank is None
    assert len(result.hits) == 2
    assert {"embed", "vector_search", "total"} <= result.timings_ms.keys()


async def test_keyword_mode_does_not_call_the_embedder(
    uow: UnitOfWorkFactory, embedder: FakeEmbedder, collection: Collection
) -> None:
    result = await SearchCollection(uow, embedder).execute(
        SearchQuery(collection.id, "etag", mode=SearchMode.KEYWORD)
    )

    assert [h.match.text for h in result.hits] == ["etag validates caches"]
    assert embedder.query_calls == []
    assert "embed" not in result.timings_ms


async def test_hybrid_mode_fuses_both_and_reports_both_ranks(
    uow: UnitOfWorkFactory, embedder: FakeEmbedder, collection: Collection
) -> None:
    result = await SearchCollection(uow, embedder).execute(
        SearchQuery(collection.id, "ingress routes http", mode=SearchMode.HYBRID)
    )

    top = result.hits[0]
    assert top.match.text == "ingress routes http"
    assert top.vector_rank == 1 and top.keyword_rank == 1
    assert top.score == pytest.approx(2 / 61)  # RRF: 1st in both lists
    assert embedder.query_calls == ["ingress routes http"]


async def test_search_rejects_embedding_model_mismatch(
    uow: UnitOfWorkFactory, collection: Collection
) -> None:
    other = FakeEmbedder(EmbeddingSpec(model="other", dimensions=8))
    with pytest.raises(ConflictError, match="other"):
        await SearchCollection(uow, other).execute(SearchQuery(collection.id, "pods"))

    # Keyword search doesn't involve vectors, so it still works.
    result = await SearchCollection(uow, other).execute(
        SearchQuery(collection.id, "pods", mode=SearchMode.KEYWORD)
    )
    assert result.hits


async def test_search_unknown_collection(uow: UnitOfWorkFactory, embedder: FakeEmbedder) -> None:
    with pytest.raises(CollectionNotFoundError):
        await SearchCollection(uow, embedder).execute(SearchQuery(new_id(), "pods"))


@pytest.mark.parametrize(
    "kwargs",
    [{"text": "  "}, {"top_k": 0}, {"top_k": 51}, {"top_k": 10, "candidates": 5}],
)
def test_search_query_validation(kwargs: dict[str, object]) -> None:
    params: dict[str, object] = {"collection_id": new_id(), "text": "pods", **kwargs}
    with pytest.raises(DomainValidationError):
        SearchQuery(**params)  # type: ignore[arg-type]


# --- Reranking ----------------------------------------------------------------
async def test_reranker_reorders_and_keeps_first_stage_rank(
    uow: UnitOfWorkFactory, embedder: FakeEmbedder, collection: Collection
) -> None:
    reranker = FakeReranker(preferred={"etag"})
    search = SearchCollection(uow, embedder, reranker)

    result = await search.execute(
        SearchQuery(collection.id, "ingress routes http", mode=SearchMode.HYBRID)
    )

    top = result.hits[0]
    assert top.match.text == "etag validates caches"  # promoted by the reranker
    assert top.rerank_score == 1.0 and top.score == 1.0
    assert top.retrieval_rank > 1  # it was not first before reranking
    assert result.reranker == "fake"
    assert "rerank" in result.timings_ms
    # The reranker reads the location (title > headings) and the passage.
    assert reranker.calls[0][1][0].text.startswith("Guide\n")


async def test_rerank_ties_keep_first_stage_order(
    uow: UnitOfWorkFactory, embedder: FakeEmbedder, collection: Collection
) -> None:
    plain = await SearchCollection(uow, embedder).execute(
        SearchQuery(collection.id, "ingress routes http")
    )
    tied = await SearchCollection(uow, embedder, FakeReranker()).execute(  # all grades 0
        SearchQuery(collection.id, "ingress routes http")
    )
    assert [h.match.chunk_id for h in tied.hits] == [h.match.chunk_id for h in plain.hits]


async def test_reranker_failure_falls_back_to_first_stage_order(
    uow: UnitOfWorkFactory, embedder: FakeEmbedder, collection: Collection
) -> None:
    result = await SearchCollection(uow, embedder, FakeReranker(fail=True)).execute(
        SearchQuery(collection.id, "ingress routes http")
    )

    assert result.hits  # the request still succeeds
    assert result.reranker is None
    assert result.rerank_error == "simulated reranker outage"
    assert all(h.rerank_score is None for h in result.hits)


async def test_rerank_can_be_disabled_per_request(
    uow: UnitOfWorkFactory, embedder: FakeEmbedder, collection: Collection
) -> None:
    reranker = FakeReranker(preferred={"etag"})
    result = await SearchCollection(uow, embedder, reranker).execute(
        SearchQuery(collection.id, "ingress routes http", rerank=False)
    )
    assert reranker.calls == []
    assert result.reranker is None


async def test_rerank_requested_without_reranker_is_rejected(
    uow: UnitOfWorkFactory, embedder: FakeEmbedder, collection: Collection
) -> None:
    with pytest.raises(DomainValidationError, match="not configured"):
        await SearchCollection(uow, embedder).execute(
            SearchQuery(collection.id, "pods", rerank=True)
        )


async def test_rerank_depth_limits_candidates_but_covers_top_k(
    uow: UnitOfWorkFactory, embedder: FakeEmbedder, collection: Collection
) -> None:
    reranker = FakeReranker()
    search = SearchCollection(uow, embedder, reranker, rerank_depth=1)

    result = await search.execute(SearchQuery(collection.id, "pods", top_k=2))

    assert len(reranker.calls[0][1]) == 2  # max(depth=1, top_k=2)
    assert len(result.hits) == 2
