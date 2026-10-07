"""One-shot RAG: retrieve → assemble context → generate a cited answer.

    question → search (hybrid + rerank) → numbered sources → LLM → answer with [n]

"One-shot" because retrieval happens exactly once, with the user's question
as the query. The agent (/agent/ask) instead decides what to search for, and
how often.
"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from uuid import UUID

from app.application.ports.chat import (
    ChatMessage,
    ChatModel,
    CompletionDone,
    TextDelta,
    TokenUsage,
)
from app.application.prompts import answer as prompts
from app.application.use_cases.answering.citations import (
    check_citations,
    normalize_citation_marks,
)
from app.application.use_cases.answering.context import ContextBuilder, ContextSource
from app.application.use_cases.retrieval import (
    SearchCollection,
    SearchMode,
    SearchQuery,
    SearchResult,
)
from app.domain.exceptions import DomainValidationError

MAX_QUESTION_LENGTH = 2000


@dataclass(frozen=True, slots=True)
class AskQuery:
    collection_id: UUID
    question: str
    mode: SearchMode = SearchMode.HYBRID
    top_k: int = 8
    rerank: bool | None = None

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise DomainValidationError("Question must not be empty")
        if len(self.question) > MAX_QUESTION_LENGTH:
            raise DomainValidationError(
                f"Question must be at most {MAX_QUESTION_LENGTH} characters"
            )


# --- Events (streamed in this order: SourcesFound, AnswerDelta*, AnswerCompleted)
@dataclass(frozen=True, slots=True)
class SourcesFound:
    sources: list[ContextSource]
    search: SearchResult


@dataclass(frozen=True, slots=True)
class AnswerDelta:
    text: str


@dataclass(frozen=True, slots=True)
class AnswerCompleted:
    answer: str
    cited: list[int]  # source numbers actually cited
    invalid_citations: list[int]  # cited numbers that match no source
    model: str | None  # None when no LLM call was needed
    usage: TokenUsage | None
    timings_ms: dict[str, float]


AskEvent = SourcesFound | AnswerDelta | AnswerCompleted


@dataclass(frozen=True, slots=True)
class AskResult:
    question: str
    answer: str
    sources: list[ContextSource]
    cited: list[int]
    invalid_citations: list[int]
    model: str | None
    usage: TokenUsage | None
    search: SearchResult
    timings_ms: dict[str, float] = field(default_factory=dict)


class AskQuestion:
    def __init__(self, search: SearchCollection, chat: ChatModel, context: ContextBuilder) -> None:
        self._search = search
        self._chat = chat
        self._context = context

    async def stream(self, query: AskQuery) -> AsyncIterator[AskEvent]:
        started = time.perf_counter()
        search = await self._search.execute(
            SearchQuery(
                collection_id=query.collection_id,
                text=query.question,
                mode=query.mode,
                top_k=query.top_k,
                rerank=query.rerank,
            )
        )
        sources = self._context.build(search.hits)
        yield SourcesFound(sources=sources, search=search)

        timings = {"search": search.timings_ms["total"]}
        if not sources:
            # Nothing relevant: answering anyway is how hallucinations happen.
            yield AnswerDelta(prompts.NO_SOURCES_ANSWER)
            timings["total"] = _ms(started)
            yield AnswerCompleted(prompts.NO_SOURCES_ANSWER, [], [], None, None, timings)
            return

        messages = [
            ChatMessage("system", prompts.SYSTEM_PROMPT),
            ChatMessage("user", prompts.user_prompt(query.question, sources)),
        ]
        parts: list[str] = []
        usage: TokenUsage | None = None
        generation_started = time.perf_counter()
        async for event in self._chat.stream(messages):
            match event:
                case TextDelta(text=raw):
                    if not parts:
                        timings["first_token"] = _ms(started)
                    text = normalize_citation_marks(raw)
                    parts.append(text)
                    yield AnswerDelta(text)
                case CompletionDone(usage=final_usage):
                    usage = final_usage

        answer = "".join(parts).strip()
        citations = check_citations(answer, len(sources))
        timings["generate"] = _ms(generation_started)
        timings["total"] = _ms(started)
        yield AnswerCompleted(
            answer=answer,
            cited=citations.cited,
            invalid_citations=citations.invalid,
            model=self._chat.model,
            usage=usage,
            timings_ms=timings,
        )

    async def execute(self, query: AskQuery) -> AskResult:
        """The same pipeline, collected into one result (for non-streaming clients)."""
        sources: SourcesFound | None = None
        completed: AnswerCompleted | None = None
        async for event in self.stream(query):
            if isinstance(event, SourcesFound):
                sources = event
            elif isinstance(event, AnswerCompleted):
                completed = event
        if sources is None or completed is None:  # pragma: no cover - stream() guarantees both
            raise RuntimeError("answer stream ended early")
        return AskResult(
            question=query.question,
            answer=completed.answer,
            sources=sources.sources,
            cited=completed.cited,
            invalid_citations=completed.invalid_citations,
            model=completed.model,
            usage=completed.usage,
            search=sources.search,
            timings_ms=completed.timings_ms,
        )


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


__all__ = [
    "AnswerCompleted",
    "AnswerDelta",
    "AskEvent",
    "AskQuery",
    "AskQuestion",
    "AskResult",
    "ContextBuilder",
    "ContextSource",
    "SourcesFound",
]
