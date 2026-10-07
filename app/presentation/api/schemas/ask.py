from uuid import UUID

from pydantic import BaseModel, Field

from app.application.ports.chat import TokenUsage
from app.application.use_cases.answering import MAX_QUESTION_LENGTH as MAX_QUESTION
from app.application.use_cases.answering import (
    AnswerCompleted,
    AskResult,
    ContextSource,
)
from app.application.use_cases.retrieval import MAX_TOP_K, SearchMode


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION,
        examples=["How do I make sure a CDN never serves an old version of my JavaScript?"],
    )
    mode: SearchMode = SearchMode.HYBRID
    top_k: int = Field(default=8, ge=1, le=MAX_TOP_K, description="Chunks retrieved")
    rerank: bool | None = Field(default=None, description="Default: on if a reranker is set")
    stream: bool = Field(
        default=False,
        description="Stream Server-Sent Events (`sources`, `token`…, `done`) instead of JSON",
    )


class SourceResponse(BaseModel):
    number: int = Field(description="Cited in the answer as [number]")
    document_id: UUID
    document_title: str
    heading_path: list[str]
    page: int | None
    chunk_ids: list[UUID]
    text: str

    @classmethod
    def from_domain(cls, source: ContextSource) -> "SourceResponse":
        return cls(
            number=source.number,
            document_id=source.document_id,
            document_title=source.document_title,
            heading_path=list(source.heading_path),
            page=source.page,
            chunk_ids=list(source.chunk_ids),
            text=source.text,
        )


class UsageResponse(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse] = Field(description="Everything the model was given")
    cited: list[int] = Field(description="Source numbers cited in the answer")
    invalid_citations: list[int] = Field(description="Cited numbers that match no source")
    model: str | None
    usage: UsageResponse | None
    reranker: str | None
    timings_ms: dict[str, float]

    @classmethod
    def from_domain(cls, result: AskResult) -> "AskResponse":
        return cls(
            question=result.question,
            answer=result.answer,
            sources=[SourceResponse.from_domain(s) for s in result.sources],
            cited=result.cited,
            invalid_citations=result.invalid_citations,
            model=result.model,
            usage=_usage(result.usage),
            reranker=result.search.reranker,
            timings_ms=result.timings_ms,
        )


# --- Streaming event payloads ---------------------------------------------------
class SourcesEvent(BaseModel):
    sources: list[SourceResponse]
    reranker: str | None


class TokenEvent(BaseModel):
    text: str


class DoneEvent(BaseModel):
    answer: str
    cited: list[int]
    invalid_citations: list[int]
    model: str | None
    usage: UsageResponse | None
    timings_ms: dict[str, float]

    @classmethod
    def from_domain(cls, event: AnswerCompleted) -> "DoneEvent":
        return cls(
            answer=event.answer,
            cited=event.cited,
            invalid_citations=event.invalid_citations,
            model=event.model,
            usage=_usage(event.usage),
            timings_ms=event.timings_ms,
        )


def _usage(usage: TokenUsage | None) -> UsageResponse | None:
    if usage is None:
        return None
    return UsageResponse(
        prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens
    )
