"""Retrieval: find the chunks most relevant to a query within one collection.

Three modes, so their behaviour can be compared on the same question:

- vector:  embed the query, nearest chunks by cosine similarity. Finds
           paraphrases ("stale JavaScript" ≈ "old version of my JS file") but can
           miss exact tokens such as error codes, flags or product names.
- keyword: PostgreSQL full-text search. Precise on exact (stemmed) terms, blind
           to synonyms and paraphrases.
- hybrid:  both, merged with Reciprocal Rank Fusion. The recommended default.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

from app.application.ports.embeddings import EmbeddingProvider
from app.application.ports.search import ChunkMatch
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.use_cases.collections import CollectionNotFoundError
from app.domain.exceptions import ConflictError, DomainValidationError
from app.domain.services.rank_fusion import DEFAULT_RRF_K, reciprocal_rank_fusion

MAX_TOP_K = 50
MAX_CANDIDATES = 200
MAX_QUERY_LENGTH = 2000


class SearchMode(StrEnum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class SearchQuery:
    collection_id: UUID
    text: str
    mode: SearchMode = SearchMode.HYBRID
    top_k: int = 8  # results returned
    candidates: int = 50  # per retriever, before fusion

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise DomainValidationError("Query must not be empty")
        if len(self.text) > MAX_QUERY_LENGTH:
            raise DomainValidationError(f"Query must be at most {MAX_QUERY_LENGTH} characters")
        if not 1 <= self.top_k <= MAX_TOP_K:
            raise DomainValidationError(f"top_k must be between 1 and {MAX_TOP_K}")
        if not self.top_k <= self.candidates <= MAX_CANDIDATES:
            raise DomainValidationError(f"candidates must be between top_k and {MAX_CANDIDATES}")


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A result, with how each retriever saw it: useful for learning and debugging."""

    match: ChunkMatch
    score: float  # the score used for the final order (depends on the mode)
    vector_rank: int | None = None
    vector_similarity: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    mode: SearchMode
    hits: list[SearchHit]
    timings_ms: dict[str, float] = field(default_factory=dict)


class SearchCollection:
    def __init__(self, uow_factory: UnitOfWorkFactory, embedder: EmbeddingProvider) -> None:
        self._uow_factory = uow_factory
        self._embedder = embedder

    async def execute(self, query: SearchQuery) -> SearchResult:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        use_vector = query.mode in (SearchMode.VECTOR, SearchMode.HYBRID)
        use_keyword = query.mode in (SearchMode.KEYWORD, SearchMode.HYBRID)

        async with self._uow_factory() as uow:
            collection = await uow.collections.get(query.collection_id)
            if collection is None:
                raise CollectionNotFoundError(query.collection_id)
            if use_vector and collection.embedding != self._embedder.spec:
                raise ConflictError(
                    f"Collection '{collection.name}' was embedded with "
                    f"{collection.embedding.model}, but the configured embedder is "
                    f"{self._embedder.spec.model}; vector search would be meaningless"
                )

            vector_matches: list[ChunkMatch] = []
            keyword_matches: list[ChunkMatch] = []
            if use_vector:
                with _timer(timings, "embed"):
                    embedding = await self._embedder.embed_query(query.text)
                with _timer(timings, "vector_search"):
                    vector_matches = await uow.search.vector_search(
                        collection.id, embedding, query.candidates
                    )
            if use_keyword:
                with _timer(timings, "keyword_search"):
                    keyword_matches = await uow.search.keyword_search(
                        collection.id, query.text, query.candidates
                    )

        hits = _combine(query, vector_matches, keyword_matches)
        timings["total"] = _ms(started)
        return SearchResult(query=query.text, mode=query.mode, hits=hits, timings_ms=timings)


def _combine(
    query: SearchQuery, vector: list[ChunkMatch], keyword: list[ChunkMatch]
) -> list[SearchHit]:
    by_id = {m.chunk_id: m for m in [*keyword, *vector]}
    vector_rank = {m.chunk_id: (rank, m.score) for rank, m in enumerate(vector, start=1)}
    keyword_rank = {m.chunk_id: (rank, m.score) for rank, m in enumerate(keyword, start=1)}

    if query.mode is SearchMode.HYBRID:
        fused = reciprocal_rank_fusion(
            [[m.chunk_id for m in vector], [m.chunk_id for m in keyword]], k=DEFAULT_RRF_K
        )
    else:
        source = vector if query.mode is SearchMode.VECTOR else keyword
        fused = [(m.chunk_id, m.score) for m in source]

    hits = []
    for chunk_id, score in fused[: query.top_k]:
        v = vector_rank.get(chunk_id)
        k = keyword_rank.get(chunk_id)
        hits.append(
            SearchHit(
                match=by_id[chunk_id],
                score=score,
                vector_rank=v[0] if v else None,
                vector_similarity=v[1] if v else None,
                keyword_rank=k[0] if k else None,
                keyword_score=k[1] if k else None,
            )
        )
    return hits


@contextmanager
def _timer(timings: dict[str, float], name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = _ms(started)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


__all__ = [
    "SearchCollection",
    "SearchHit",
    "SearchMode",
    "SearchQuery",
    "SearchResult",
]
