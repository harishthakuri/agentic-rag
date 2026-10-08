"""Agentic search: the model decides what to search for, how often, and when to stop.

    question ─► LLM ─► tool call? ──yes──► run tool, append result ─┐
                 ▲                                                  │
                 └──────────────────────────────────────────────────┘
                 └─no──► final answer (cited [n])

Compared with one-shot RAG (/ask), the agent can split a question into several
searches, rephrase when results are poor, and read around a passage. This costs
more LLM calls and time. Guards keep it bounded:

- max tool calls: then tools are withdrawn and the model must answer
- prompt size: the same, if the conversation grows too large
- wall-clock timeout: the run fails cleanly
- tool errors (bad arguments, unknown tools, repeated searches) are returned to
  the model as text, so it can correct itself instead of crashing the run
- grounding: a final answer that cites none of the passages (and doesn't decline)
  is withheld, because it came from the model's memory, not the documents

Every run and step is stored (see AgentRun), so you can replay how it reasoned.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.application.ports.chat import (
    ChatMessage,
    ChatModel,
    CompletionDone,
    TextDelta,
    TokenUsage,
    ToolCall,
)
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.prompts import agent as prompts
from app.application.prompts.answer import UNGROUNDED_ANSWER
from app.application.use_cases.answering import AnswerDelta
from app.application.use_cases.answering.citations import (
    check_citations,
    is_grounded,
    normalize_citation_marks,
)
from app.application.use_cases.collections import CollectionNotFoundError
from app.application.use_cases.retrieval import SearchCollection
from app.domain.exceptions import DomainValidationError, NotFoundError
from app.domain.models import AgentRun, AgentStep

from .tools import TOOLS, AgentToolbox, Source, SourceRegistry, ToolOutcome, parse_arguments

logger = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 2000


class AgentTimeoutError(Exception):
    """The run exceeded its wall-clock budget."""


@dataclass(frozen=True, slots=True)
class AgentQuery:
    collection_id: UUID
    question: str
    max_tool_calls: int | None = None  # None: the configured default

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise DomainValidationError("Question must not be empty")
        if len(self.question) > MAX_QUESTION_LENGTH:
            raise DomainValidationError(
                f"Question must be at most {MAX_QUESTION_LENGTH} characters"
            )
        if self.max_tool_calls is not None and not 1 <= self.max_tool_calls <= 12:
            raise DomainValidationError("max_tool_calls must be between 1 and 12")


# --- Events (RunStarted, then ToolCalled/ToolReturned pairs, AnswerDelta*, AgentCompleted)
@dataclass(frozen=True, slots=True)
class RunStarted:
    run_id: UUID


@dataclass(frozen=True, slots=True)
class ToolCalled:
    step: int
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolReturned:
    step: int
    tool: str
    summary: dict[str, Any]
    latency_ms: float


@dataclass(frozen=True, slots=True)
class AgentCompleted:
    run_id: UUID
    answer: str
    sources: list[Source]  # every passage the agent saw, numbered
    cited: list[int]
    invalid_citations: list[int]
    tool_calls: int
    model: str
    usage: TokenUsage
    timings_ms: dict[str, float] = field(default_factory=dict)
    withheld: bool = False  # the model's answer cited nothing and was replaced


AgentEvent = RunStarted | ToolCalled | ToolReturned | AnswerDelta | AgentCompleted


class AgentAsk:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        search: SearchCollection,
        chat: ChatModel,
        *,
        max_tool_calls: int = 6,
        max_prompt_tokens: int = 24_000,
        timeout_seconds: float = 180.0,
        search_top_k: int = 5,
        rerank: bool = False,
    ) -> None:
        self._uow_factory = uow_factory
        self._search = search
        self._chat = chat
        self._max_tool_calls = max_tool_calls
        self._max_prompt_tokens = max_prompt_tokens
        self._timeout = timeout_seconds
        self._search_top_k = search_top_k
        self._rerank = rerank

    async def stream(self, query: AgentQuery) -> AsyncIterator[AgentEvent]:
        started = time.perf_counter()
        async with self._uow_factory() as uow:
            if await uow.collections.get(query.collection_id) is None:
                raise CollectionNotFoundError(query.collection_id)
            run = AgentRun(
                collection_id=query.collection_id, question=query.question, model=self._chat.model
            )
            await uow.agent_runs.add(run)
            await uow.commit()
        yield RunStarted(run.id)

        try:
            async with asyncio.timeout(self._timeout):
                async for event in self._loop(run, query, started):
                    yield event
        except TimeoutError as exc:
            await self._fail(run, f"timed out after {self._timeout:.0f}s", started)
            raise AgentTimeoutError(f"Agent run exceeded {self._timeout:.0f}s") from exc
        except (Exception, asyncio.CancelledError) as exc:  # incl. client disconnects
            await self._fail(run, f"{type(exc).__name__}: {exc}"[:1000], started)
            raise

    async def _loop(
        self, run: AgentRun, query: AgentQuery, started: float
    ) -> AsyncIterator[AgentEvent]:
        limit = query.max_tool_calls or self._max_tool_calls
        registry = SourceRegistry()
        toolbox = AgentToolbox(
            query.collection_id,
            self._search,
            self._uow_factory,
            registry,
            top_k=self._search_top_k,
            rerank=self._rerank,
        )
        messages = [
            ChatMessage("system", prompts.system_prompt(limit)),
            ChatMessage("user", query.question),
        ]
        usage = TokenUsage(0, 0)
        tool_calls = 0
        tools_withdrawn = False
        timings: dict[str, float] = {"llm": 0.0, "tools": 0.0}

        while True:
            if not tools_withdrawn and (
                tool_calls >= limit or _estimate_tokens(messages) > self._max_prompt_tokens
            ):
                tools_withdrawn = True
                messages.append(ChatMessage("user", prompts.FINAL_ANSWER_NUDGE))

            parts: list[str] = []
            requested: tuple[ToolCall, ...] = ()
            llm_started = time.perf_counter()
            async for event in self._chat.stream(messages, () if tools_withdrawn else TOOLS):
                match event:
                    case TextDelta(text=raw):
                        text = normalize_citation_marks(raw)
                        parts.append(text)
                        yield AnswerDelta(text)
                    case CompletionDone(usage=turn_usage, tool_calls=calls):
                        requested = calls
                        if turn_usage:
                            usage = usage + turn_usage
            timings["llm"] += _ms(llm_started)

            if not requested:
                answer = "".join(parts).strip()
                break

            messages.append(ChatMessage("assistant", "".join(parts), tool_calls=requested))
            for call in requested:
                if tool_calls >= limit:  # several calls in one turn can overshoot the limit
                    messages.append(
                        ChatMessage("tool", "Error: tool-call limit reached.", tool_call_id=call.id)
                    )
                    continue
                tool_calls += 1
                yield ToolCalled(tool_calls, call.name, parse_arguments(call))
                tool_started = time.perf_counter()
                outcome = await toolbox.run(call)
                latency = _ms(tool_started)
                timings["tools"] += latency
                messages.append(ChatMessage("tool", outcome.content, tool_call_id=call.id))
                await self._record_step(run, tool_calls, call, outcome, latency)
                yield ToolReturned(tool_calls, call.name, outcome.summary, latency)

        citations = check_citations(answer, len(registry))
        # Exception: a question about the collection itself ("which documents are
        # there?") is answered from list_documents, which has no passages to cite.
        about_collection = len(registry) == 0 and toolbox.listed_documents
        withheld = not is_grounded(answer, citations) and not about_collection
        if withheld:
            logger.warning(
                "Agent run %s: withheld an answer that cited no sources: %.500s", run.id, answer
            )
            answer = UNGROUNDED_ANSWER  # invalid citations stay reported
        timings["total"] = _ms(started)
        run.record_usage(usage.prompt_tokens, usage.completion_tokens)
        run.succeed(answer, timings["total"])
        async with self._uow_factory() as uow:
            await uow.agent_runs.update(run)
            await uow.commit()
        yield AgentCompleted(
            run_id=run.id,
            answer=answer,
            sources=registry.all(),
            cited=citations.cited,
            invalid_citations=citations.invalid,
            tool_calls=tool_calls,
            model=self._chat.model,
            usage=usage,
            timings_ms=timings,
            withheld=withheld,
        )

    async def _record_step(
        self, run: AgentRun, number: int, call: ToolCall, outcome: ToolOutcome, latency: float
    ) -> None:
        run.step_count = number
        async with self._uow_factory() as uow:
            await uow.agent_runs.add_step(
                AgentStep(
                    run_id=run.id,
                    number=number,
                    tool=call.name,
                    arguments=outcome.arguments,
                    result=outcome.summary,
                    latency_ms=latency,
                )
            )
            await uow.agent_runs.update(run)
            await uow.commit()

    async def _fail(self, run: AgentRun, error: str, started: float) -> None:
        logger.warning("Agent run %s failed: %s", run.id, error)
        try:
            run.fail(error, _ms(started))
            async with self._uow_factory() as uow:
                await uow.agent_runs.update(run)
                await uow.commit()
        except Exception:  # never mask the original error
            logger.exception("Could not record failure of agent run %s", run.id)

    async def execute(self, query: AgentQuery) -> tuple[AgentCompleted, list[AgentEvent]]:
        """Run to completion; returns the result and every event (for JSON clients)."""
        events = [event async for event in self.stream(query)]
        completed = events[-1]
        if not isinstance(completed, AgentCompleted):  # pragma: no cover
            raise RuntimeError("agent stream ended without a result")
        return completed, events


def _estimate_tokens(messages: list[ChatMessage]) -> int:
    """Rough size of the next prompt: characters / 4 (tokens ≈ 4 chars of English).
    Deterministic and model-independent, which is all a guard needs."""
    return sum(len(m.content) for m in messages) // 4


@dataclass(frozen=True, slots=True)
class AgentRunDetails:
    run: AgentRun
    steps: list[AgentStep]


class GetAgentRun:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, run_id: UUID) -> AgentRunDetails:
        async with self._uow_factory() as uow:
            run = await uow.agent_runs.get(run_id)
            if run is None:
                raise NotFoundError(f"Agent run {run_id} not found")
            return AgentRunDetails(run=run, steps=await uow.agent_runs.steps(run_id))


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


__all__ = [
    "AgentAsk",
    "AgentCompleted",
    "AgentEvent",
    "AgentQuery",
    "AgentRunDetails",
    "AgentTimeoutError",
    "GetAgentRun",
    "RunStarted",
    "ToolCalled",
    "ToolReturned",
]
