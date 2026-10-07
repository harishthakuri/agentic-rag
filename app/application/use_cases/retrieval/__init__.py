"""Retrieval: find the chunks most relevant to a query within one collection.

Three modes, so their behaviour can be compared on the same question:

- vector:  embed the query, nearest chunks by cosine similarity. Finds
           paraphrases ("stale JavaScript" ≈ "old version of my JS file") but can
           miss exact tokens such as error codes, flags or product names.
- keyword: PostgreSQL full-text search. Precise on exact (stemmed) terms, blind
           to synonyms and paraphrases.
- hybrid:  both, merged with Reciprocal Rank Fusion. The recommended default.

Optionally, a reranker then re-orders the top candidates of any mode by reading
each one next to the query (see ports/reranking.py).
"""

import dataclasses
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

from app.application.ports.embeddings import EmbeddingProvider
from app.application.ports.reranking import RerankCandidate, Reranker, RerankError
from app.application.ports.search import ChunkMatch
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.use_cases.collections import CollectionNotFoundError
from app.domain.exceptions import ConflictError, DomainValidationError
from app.domain.services.rank_fusion import DEFAULT_RRF_K, reciprocal_rank_fusion

MAX_TOP_K = 50
MAX_CANDIDATES = 200
MAX_QUERY_LENGTH = 2000

logger = logging.getLogger(__name__)


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
    rerank: bool | None = None  # None: rerank if a reranker is configured

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
    score: float  # the score used for the final order (depends on mode and reranking)
    retrieval_rank: int  # position after the first stage (before any reranking)
    rerank_score: float | None = None
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
    reranker: str | None = None  # set when the hits were reranked
    rerank_error: str | None = None  # set when reranking failed and was skipped


class SearchCollection:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        embedder: EmbeddingProvider,
        reranker: Reranker | None = None,
        *,
        rerank_depth: int = 10,
    ) -> None:
        self._uow_factory = uow_factory
        self._embedder = embedder
        self._reranker = reranker
        self._rerank_depth = rerank_depth  # how many first-stage candidates to rerank

    async def execute(self, query: SearchQuery) -> SearchResult:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        use_vector = query.mode in (SearchMode.VECTOR, SearchMode.HYBRID)
        use_keyword = query.mode in (SearchMode.KEYWORD, SearchMode.HYBRID)
        rerank = self._reranker is not None if query.rerank is None else query.rerank
        if rerank and self._reranker is None:
            raise DomainValidationError("Reranking is not configured (RERANKER=none)")

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

        hits = _first_stage(query, vector_matches, keyword_matches)
        reranker_name: str | None = None
        rerank_error: str | None = None
        if rerank and self._reranker is not None and hits:
            with _timer(timings, "rerank"):
                try:
                    hits = await self._rerank(
                        self._reranker, query.text, hits, max(self._rerank_depth, query.top_k)
                    )
                    reranker_name = self._reranker.name
                except RerankError as exc:
                    logger.warning("Reranking failed, using first-stage order: %s", exc)
                    rerank_error = str(exc)

        timings["total"] = _ms(started)
        return SearchResult(
            query=query.text,
            mode=query.mode,
            hits=hits[: query.top_k],
            timings_ms=timings,
            reranker=reranker_name,
            rerank_error=rerank_error,
        )

    async def _rerank(
        self, reranker: Reranker, query: str, hits: list[SearchHit], depth: int
    ) -> list[SearchHit]:
        pool = hits[:depth]
        scores = await reranker.rerank(query, [_candidate(h.match) for h in pool])
        by_id = {s.id: s.score for s in scores}
        if by_id.keys() != {h.match.chunk_id for h in pool}:
            raise RerankError("reranker did not score every candidate")
        rescored = [
            dataclasses.replace(
                h, score=by_id[h.match.chunk_id], rerank_score=by_id[h.match.chunk_id]
            )
            for h in pool
        ]
        # Ties (common with coarse LLM grades) keep the first-stage order.
        rescored.sort(key=lambda h: (-h.score, h.retrieval_rank))
        return rescored


def _first_stage(
    query: SearchQuery, vector: list[ChunkMatch], keyword: list[ChunkMatch]
) -> list[SearchHit]:
    """All first-stage candidates in order (not yet cut to top_k: reranking may follow)."""
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
    for position, (chunk_id, score) in enumerate(fused, start=1):
        v = vector_rank.get(chunk_id)
        k = keyword_rank.get(chunk_id)
        hits.append(
            SearchHit(
                match=by_id[chunk_id],
                score=score,
                retrieval_rank=position,
                vector_rank=v[0] if v else None,
                vector_similarity=v[1] if v else None,
                keyword_rank=k[0] if k else None,
                keyword_score=k[1] if k else None,
            )
        )
    return hits


def _candidate(match: ChunkMatch) -> RerankCandidate:
    location = " > ".join((match.document_title, *match.heading_path))
    return RerankCandidate(id=match.chunk_id, text=f"{location}\n{match.text}")


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
