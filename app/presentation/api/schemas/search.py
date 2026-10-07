from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.application.use_cases.retrieval import (
    MAX_CANDIDATES,
    MAX_QUERY_LENGTH,
    MAX_TOP_K,
    SearchHit,
    SearchMode,
    SearchResult,
)


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        examples=["Why does my filtered vector search return fewer rows than the LIMIT?"],
    )
    mode: SearchMode = Field(
        default=SearchMode.HYBRID,
        description="`vector` (semantic), `keyword` (full-text) or `hybrid` (both, fused with RRF)",
    )
    top_k: int = Field(default=8, ge=1, le=MAX_TOP_K, description="Results to return")
    candidates: int = Field(
        default=50,
        ge=1,
        le=MAX_CANDIDATES,
        description="Candidates fetched from each retriever before fusion",
    )
    rerank: bool | None = Field(
        default=None,
        description="Rerank the top candidates (default: on when a reranker is configured)",
    )

    @model_validator(mode="after")
    def _candidates_cover_top_k(self) -> "SearchRequest":
        if self.candidates < self.top_k:
            raise ValueError("candidates must be at least top_k")
        return self


class RetrieverRank(BaseModel):
    rank: int
    score: float


class SearchHitResponse(BaseModel):
    chunk_id: UUID
    document_id: UUID
    document_title: str
    heading_path: list[str]
    page: int | None
    text: str
    score: float = Field(
        description="Final score: rerank grade, else similarity, keyword score or RRF score"
    )
    retrieval_rank: int = Field(description="Position after the first stage, before reranking")
    rerank_score: float | None
    vector: RetrieverRank | None = Field(description="Position in the vector results, if any")
    keyword: RetrieverRank | None = Field(description="Position in the keyword results, if any")

    @classmethod
    def from_domain(cls, hit: SearchHit) -> "SearchHitResponse":
        m = hit.match
        return cls(
            chunk_id=m.chunk_id,
            document_id=m.document_id,
            document_title=m.document_title,
            heading_path=list(m.heading_path),
            page=m.page,
            text=m.text,
            score=round(hit.score, 6),
            retrieval_rank=hit.retrieval_rank,
            rerank_score=hit.rerank_score,
            vector=_rank(hit.vector_rank, hit.vector_similarity),
            keyword=_rank(hit.keyword_rank, hit.keyword_score),
        )


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    hits: list[SearchHitResponse]
    timings_ms: dict[str, float]
    reranker: str | None = Field(description="Reranker used, if the hits were reranked")
    rerank_error: str | None = Field(
        description="Why reranking was skipped (hits are then in first-stage order)"
    )

    @classmethod
    def from_domain(cls, result: SearchResult) -> "SearchResponse":
        return cls(
            query=result.query,
            mode=result.mode,
            hits=[SearchHitResponse.from_domain(h) for h in result.hits],
            timings_ms=result.timings_ms,
            reranker=result.reranker,
            rerank_error=result.rerank_error,
        )


def _rank(rank: int | None, score: float | None) -> RetrieverRank | None:
    if rank is None or score is None:
        return None
    return RetrieverRank(rank=rank, score=round(score, 6))
