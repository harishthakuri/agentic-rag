from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import UUID

from app.domain.exceptions import InvalidStateTransitionError
from app.domain.models.entity import Entity
from app.domain.value_objects import new_id

_BASE_BACKOFF = timedelta(seconds=30)


class JobStatus(StrEnum):
    QUEUED = "queued"  # waiting for a worker (possibly until `run_after`)
    RUNNING = "running"  # claimed by `locked_by`
    SUCCEEDED = "succeeded"
    FAILED = "failed"  # no attempts left, or a permanent error


@dataclass(eq=False, kw_only=True, slots=True)
class IngestionJob(Entity):
    """Background work to parse, chunk and embed one document.

    Workers claim jobs atomically (see IngestionJobRepository.claim_next), which
    sets status=RUNNING and increments `attempts`. A failed attempt is retried
    with exponential backoff until `max_attempts` is reached.
    """

    id: UUID = field(default_factory=new_id)
    document_id: UUID
    status: JobStatus = JobStatus.QUEUED
    attempts: int = 0
    max_attempts: int = 3
    last_error: str | None = None
    run_after: datetime = field(default_factory=lambda: datetime.now(UTC))
    locked_at: datetime | None = None
    locked_by: str | None = None
    finished_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def for_document(cls, document_id: UUID, *, max_attempts: int = 3) -> "IngestionJob":
        return cls(document_id=document_id, max_attempts=max_attempts)

    def succeed(self) -> None:
        self._require_running()
        now = datetime.now(UTC)
        self.status = JobStatus.SUCCEEDED
        self.finished_at = now
        self._release(now)

    def fail(self, error: str, *, retryable: bool = True) -> bool:
        """Record a failed attempt. Returns True if the job will be retried."""
        self._require_running()
        now = datetime.now(UTC)
        self.last_error = error
        self._release(now)
        if retryable and self.attempts < self.max_attempts:
            self.status = JobStatus.QUEUED
            # 30s, 60s, 120s, ...: give a struggling dependency time to recover.
            self.run_after = now + _BASE_BACKOFF * 2 ** max(self.attempts - 1, 0)
            return True
        self.status = JobStatus.FAILED
        self.finished_at = now
        return False

    def _require_running(self) -> None:
        if self.status is not JobStatus.RUNNING:
            raise InvalidStateTransitionError(f"Job {self.id} is '{self.status}', not running")

    def _release(self, now: datetime) -> None:
        self.locked_at = None
        self.locked_by = None
        self.updated_at = now
