from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.application.use_cases.agent import (
    MAX_QUESTION_LENGTH,
    AgentCompleted,
    AgentRunDetails,
    ToolCalled,
    ToolReturned,
)
from app.application.use_cases.agent.tools import Source
from app.domain.models import RunStatus
from app.presentation.api.schemas.ask import UsageResponse


class AgentAskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        examples=["Compare HNSW and IVFFlat indexes: when should I use each?"],
    )
    max_tool_calls: int | None = Field(
        default=None, ge=1, le=12, description="Tool-call budget (default from settings)"
    )
    stream: bool = Field(
        default=False,
        description="Stream Server-Sent Events (`run`, `tool_call`, `tool_result`, `token`, "
        "`done`) instead of JSON",
    )


class AgentSourceResponse(BaseModel):
    number: int
    chunk_id: UUID
    document_id: UUID
    document_title: str
    heading_path: list[str]
    page: int | None
    text: str
    cited: bool

    @classmethod
    def from_domain(cls, source: Source, cited: set[int]) -> "AgentSourceResponse":
        m = source.match
        return cls(
            number=source.number,
            chunk_id=m.chunk_id,
            document_id=m.document_id,
            document_title=m.document_title,
            heading_path=list(m.heading_path),
            page=m.page,
            text=m.text,
            cited=source.number in cited,
        )


class ToolCallEvent(BaseModel):
    step: int
    tool: str
    arguments: dict[str, Any]

    @classmethod
    def from_domain(cls, event: ToolCalled) -> "ToolCallEvent":
        return cls(step=event.step, tool=event.tool, arguments=event.arguments)


class ToolResultEvent(BaseModel):
    step: int
    tool: str
    result: dict[str, Any] = Field(description="Sources found (number, location), or an error")
    latency_ms: float

    @classmethod
    def from_domain(cls, event: ToolReturned) -> "ToolResultEvent":
        return cls(
            step=event.step, tool=event.tool, result=event.summary, latency_ms=event.latency_ms
        )


class AgentAskResponse(BaseModel):
    run_id: UUID
    answer: str
    sources: list[AgentSourceResponse] = Field(description="Every passage the agent read")
    cited: list[int]
    invalid_citations: list[int]
    steps: list[ToolCallEvent]
    tool_calls: int
    model: str
    usage: UsageResponse
    timings_ms: dict[str, float]
    answer_withheld: bool = Field(
        description="True when the model's answer cited no sources and was replaced"
    )

    @classmethod
    def from_domain(cls, result: AgentCompleted, steps: list[ToolCallEvent]) -> "AgentAskResponse":
        cited = set(result.cited)
        return cls(
            run_id=result.run_id,
            answer=result.answer,
            sources=[AgentSourceResponse.from_domain(s, cited) for s in result.sources],
            cited=result.cited,
            invalid_citations=result.invalid_citations,
            steps=steps,
            tool_calls=result.tool_calls,
            model=result.model,
            usage=UsageResponse(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
            ),
            timings_ms=result.timings_ms,
            answer_withheld=result.withheld,
        )


class AgentStepResponse(BaseModel):
    number: int
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    latency_ms: float
    created_at: datetime


class AgentRunResponse(BaseModel):
    id: UUID
    collection_id: UUID
    question: str
    model: str
    status: RunStatus
    answer: str | None
    error: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float | None
    created_at: datetime
    finished_at: datetime | None
    steps: list[AgentStepResponse]

    @classmethod
    def from_domain(cls, details: AgentRunDetails) -> "AgentRunResponse":
        run = details.run
        return cls(
            id=run.id,
            collection_id=run.collection_id,
            question=run.question,
            model=run.model,
            status=run.status,
            answer=run.answer,
            error=run.error,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            latency_ms=run.latency_ms,
            created_at=run.created_at,
            finished_at=run.finished_at,
            steps=[
                AgentStepResponse(
                    number=s.number,
                    tool=s.tool,
                    arguments=s.arguments,
                    result=s.result,
                    latency_ms=s.latency_ms,
                    created_at=s.created_at,
                )
                for s in details.steps
            ],
        )
