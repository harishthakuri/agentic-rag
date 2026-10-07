from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, Float, ForeignKey, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.orm.base import Base


class AgentRunEntity(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        CheckConstraint("status IN ('running', 'succeeded', 'failed')", name="status_valid"),
        Index("ix_agent_runs_collection_id_created_at", "collection_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    collection_id: Mapped[UUID] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"))
    question: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20))
    answer: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    step_count: Mapped[int] = mapped_column(server_default=text("0"))
    prompt_tokens: Mapped[int] = mapped_column(server_default=text("0"))
    completion_tokens: Mapped[int] = mapped_column(server_default=text("0"))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    finished_at: Mapped[datetime | None]


class AgentStepEntity(Base):
    __tablename__ = "agent_steps"
    __table_args__ = (Index("uq_agent_steps_run_id_number", "run_id", "number", unique=True),)

    id: Mapped[UUID] = mapped_column(primary_key=True)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"))
    number: Mapped[int]
    tool: Mapped[str] = mapped_column(String(100))
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONB)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB)
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
