from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, Index, String, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.orm.base import Base, TimestampMixin


class IngestionJobEntity(TimestampMixin, Base):
    """A unit of background work, claimed by workers with FOR UPDATE SKIP LOCKED."""

    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')", name="status_valid"
        ),
        CheckConstraint("attempts >= 0", name="attempts_non_negative"),
        # Partial index: the worker's poll query only ever looks at queued jobs.
        Index(
            "ix_ingestion_jobs_runnable",
            "run_after",
            postgresql_where=text("status = 'queued'"),
        ),
        Index("ix_ingestion_jobs_document_id", "document_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(20))
    attempts: Mapped[int] = mapped_column(server_default=text("0"))
    max_attempts: Mapped[int] = mapped_column(server_default=text("3"))
    last_error: Mapped[str | None] = mapped_column(Text)
    run_after: Mapped[datetime] = mapped_column(server_default=func.now())
    locked_at: Mapped[datetime | None]
    locked_by: Mapped[str | None] = mapped_column(String(200))
    finished_at: Mapped[datetime | None]
