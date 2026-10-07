from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from app.domain.exceptions import InvalidStateTransitionError
from app.domain.models.entity import Entity
from app.domain.value_objects import new_id


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(eq=False, kw_only=True, slots=True)
class AgentRun(Entity):
    """One agentic answer: the question, every tool call made, and the result.

    Stored so a run can be replayed and inspected: which queries the agent chose,
    what came back, how long it took and how many tokens it used.
    """

    id: UUID = field(default_factory=new_id)
    collection_id: UUID
    question: str
    model: str
    status: RunStatus = RunStatus.RUNNING
    answer: str | None = None
    error: str | None = None
    step_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def succeed(self, answer: str, latency_ms: float) -> None:
        self._finish(RunStatus.SUCCEEDED, latency_ms)
        self.answer = answer

    def fail(self, error: str, latency_ms: float) -> None:
        self._finish(RunStatus.FAILED, latency_ms)
        self.error = error

    def _finish(self, status: RunStatus, latency_ms: float) -> None:
        if self.status is not RunStatus.RUNNING:
            raise InvalidStateTransitionError(f"Agent run {self.id} already finished")
        self.status = status
        self.latency_ms = latency_ms
        self.finished_at = datetime.now(UTC)


@dataclass(eq=False, kw_only=True, slots=True)
class AgentStep(Entity):
    """One tool call within a run, with a summary of its result."""

    id: UUID = field(default_factory=new_id)
    run_id: UUID
    number: int  # 1-based order within the run
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]  # e.g. {"sources": [...]} or {"error": "..."}
    latency_ms: float
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
