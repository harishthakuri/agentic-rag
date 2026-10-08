"""One-shot RAG: retrieve → assemble context → generate a cited answer.

    question → search (hybrid + rerank) → numbered sources → LLM → answer with [n]
                                                          └─ no [n]? one retry to add them

"One-shot" because retrieval happens exactly once, with the user's question
as the query. The agent (/agent/ask) instead decides what to search for, and
how often.
"""

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from uuid import UUID

from app.application.ports.chat import (
    ChatMessage,
    ChatModel,
    ChatModelError,
    CompletionDone,
    TextDelta,
    TokenUsage,
)
from app.application.ports.telemetry import (
    ANSWERS,
    NOOP_TELEMETRY,
    STEP_DURATION,
    Span,
    Telemetry,
)
from app.application.prompts import answer as prompts
from app.application.use_cases.answering.citations import (
    check_citations,
    is_grounded,
    normalize_citation_marks,
)
from app.application.use_cases.answering.context import ContextBuilder, ContextSource
from app.application.use_cases.answering.grounding import add_missing_citations, answer_outcome
from app.application.use_cases.retrieval import (
    SearchCollection,
    SearchMode,
    SearchQuery,
    SearchResult,
)
from app.domain.exceptions import DomainValidationError

logger = logging.getLogger(__name__)

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


# --- Events (streamed in this order: SourcesFound, AnswerDelta*, [AnswerRevising],
# AnswerCompleted)
@dataclass(frozen=True, slots=True)
class SourcesFound:
    sources: list[ContextSource]
    search: SearchResult


@dataclass(frozen=True, slots=True)
class AnswerDelta:
    text: str


@dataclass(frozen=True, slots=True)
class AnswerRevising:
    """The draft cited no sources; the model is asked once to add them."""


@dataclass(frozen=True, slots=True)
class AnswerCompleted:
    answer: str
    cited: list[int]  # source numbers actually cited
    invalid_citations: list[int]  # cited numbers that match no source
    model: str | None  # None when no LLM call was needed
    usage: TokenUsage | None
    timings_ms: dict[str, float]
    # True when the model's answer cited no sources and was replaced (the streamed
    # text should then be replaced by `answer`).
    withheld: bool = False
    # True when the draft cited no sources and `answer` is the model's rewrite with
    # citations (it replaces the streamed draft too).
    revised: bool = False


AskEvent = SourcesFound | AnswerDelta | AnswerRevising | AnswerCompleted


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
    withheld: bool = False
    revised: bool = False


class AskQuestion:
    def __init__(
        self,
        search: SearchCollection,
        chat: ChatModel,
        context: ContextBuilder,
        *,
        telemetry: Telemetry = NOOP_TELEMETRY,
    ) -> None:
        self._search = search
        self._chat = chat
        self._context = context
        self._telemetry = telemetry

    async def stream(self, query: AskQuery) -> AsyncIterator[AskEvent]:
        # A streamed answer is an async generator: the span is started here and made
        # current only around awaits, never across a `yield` (see ports/telemetry.py).
        root = self._telemetry.start_span(
            "rag.ask",
            kind="chain",
            trace_name="ask",
            attributes={
                "rag.collection.id": str(query.collection_id),
                "rag.question.chars": len(query.question),
                "rag.search.mode": query.mode.value,
                "rag.search.top_k": query.top_k,
            },
        )
        root.content(input=query.question)
        try:
            async for event in self._answer(query, root):
                yield event
        except Exception as exc:
            root.fail(exc)
            raise
        finally:
            root.end()

    async def _answer(self, query: AskQuery, root: Span) -> AsyncIterator[AskEvent]:
        started = time.perf_counter()
        with root.activate():
            search = await self._search.execute(
                SearchQuery(
                    collection_id=query.collection_id,
                    text=query.question,
                    mode=query.mode,
                    top_k=query.top_k,
                    rerank=query.rerank,
                )
            )
            with self._telemetry.span("rag.context") as span:
                sources = self._context.build(search.hits)
                span.set(
                    {
                        "rag.context.hits": len(search.hits),
                        "rag.context.sources": len(sources),
                        "rag.context.tokens": sum(s.token_count for s in sources),
                    }
                )
                span.content(output=[f"[{s.number}] {s.location}" for s in sources])
        root.set({"rag.sources": len(sources), "rag.reranker": search.reranker})
        yield SourcesFound(sources=sources, search=search)

        timings = {"search": search.timings_ms["total"]}
        if not sources:
            # Nothing relevant: answering anyway is how hallucinations happen.
            yield AnswerDelta(prompts.NO_SOURCES_ANSWER)
            timings["total"] = _ms(started)
            self._finish(root, "no_sources", prompts.NO_SOURCES_ANSWER)
            yield AnswerCompleted(prompts.NO_SOURCES_ANSWER, [], [], None, None, timings)
            return

        messages = [
            ChatMessage("system", prompts.SYSTEM_PROMPT),
            ChatMessage("user", prompts.user_prompt(query.question, sources)),
        ]
        parts: list[str] = []
        usage: TokenUsage | None = None
        generation_started = time.perf_counter()
        with root.activate():  # the model's span becomes a child of rag.ask
            answer_stream = self._chat.stream(messages)
        async for event in answer_stream:
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
        timings["generate"] = _ms(generation_started)
        self._telemetry.record(STEP_DURATION, timings["generate"] / 1000, {"step": "generate"})
        citations = check_citations(answer, len(sources))
        revised = False
        needs_repair = not is_grounded(answer, citations)
        if needs_repair:
            yield AnswerRevising()
        with root.activate(), self._telemetry.span("rag.grounding", kind="guardrail") as check:
            check.set({"rag.citations.draft": len(citations.cited), "rag.repair": needs_repair})
            if needs_repair:
                repair_started = time.perf_counter()
                try:
                    repair = await add_missing_citations(self._chat, messages, answer)
                except ChatModelError:
                    logger.warning("Citation repair failed", exc_info=True)
                else:
                    if repair.usage:
                        usage = usage + repair.usage if usage else repair.usage
                    repaired = check_citations(repair.answer, len(sources))
                    if is_grounded(repair.answer, repaired):
                        logger.info("Added missing citations to a draft: %.500s", answer)
                        answer, citations, revised = repair.answer, repaired, True
                timings["repair"] = _ms(repair_started)
                self._telemetry.record(STEP_DURATION, timings["repair"] / 1000, {"step": "repair"})
            withheld = not is_grounded(answer, citations)
            if withheld:
                logger.warning("Withheld an answer that cited no sources: %.500s", answer)
                answer = prompts.UNGROUNDED_ANSWER  # invalid citations stay reported
            check.set(
                {
                    "rag.citations.cited": len(citations.cited),
                    "rag.citations.invalid": len(citations.invalid),
                    "rag.answer.revised": revised,
                    "rag.answer.withheld": withheld,
                }
            )
        timings["total"] = _ms(started)
        self._finish(
            root, answer_outcome(citations.cited, revised=revised, withheld=withheld), answer
        )
        root.set(
            {
                "rag.usage.input_tokens": usage.prompt_tokens if usage else None,
                "rag.usage.output_tokens": usage.completion_tokens if usage else None,
            }
        )
        yield AnswerCompleted(
            answer=answer,
            cited=citations.cited,
            invalid_citations=citations.invalid,
            model=self._chat.model,
            usage=usage,
            timings_ms=timings,
            withheld=withheld,
            revised=revised,
        )

    def _finish(self, root: Span, outcome: str, answer: str) -> None:
        root.set({"rag.answer.outcome": outcome})
        root.content(output=answer)
        self._telemetry.count(ANSWERS, labels={"mode": "ask", "outcome": outcome})

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
            withheld=completed.withheld,
            revised=completed.revised,
        )


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


__all__ = [
    "AnswerCompleted",
    "AnswerDelta",
    "AnswerRevising",
    "AskEvent",
    "AskQuery",
    "AskQuestion",
    "AskResult",
    "ContextBuilder",
    "ContextSource",
    "SourcesFound",
]
