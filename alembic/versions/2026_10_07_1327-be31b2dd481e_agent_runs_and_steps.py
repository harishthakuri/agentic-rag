"""agent runs and steps

Stores every agentic answer (agent_runs) and each tool call it made (agent_steps),
so runs can be replayed and inspected via GET /api/v1/agent/runs/{id}.

Revision ID: be31b2dd481e
Revises: 4b7ce4173615
Create Date: 2026-10-07 13:27:35.170687

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "be31b2dd481e"
down_revision: str | Sequence[str] | None = "4b7ce4173615"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("collection_id", sa.Uuid(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("step_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'succeeded', 'failed')", name=op.f("ck_agent_runs_status_valid")
        ),
        sa.ForeignKeyConstraint(
            ["collection_id"],
            ["rag.collections.id"],
            name=op.f("fk_agent_runs_collection_id_collections"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_runs")),
        schema="rag",
    )
    op.create_index(
        "ix_agent_runs_collection_id_created_at",
        "agent_runs",
        ["collection_id", "created_at"],
        unique=False,
        schema="rag",
    )
    op.create_table(
        "agent_steps",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("tool", sa.String(length=100), nullable=False),
        sa.Column("arguments", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["rag.agent_runs.id"],
            name=op.f("fk_agent_steps_run_id_agent_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_steps")),
        schema="rag",
    )
    op.create_index(
        "uq_agent_steps_run_id_number",
        "agent_steps",
        ["run_id", "number"],
        unique=True,
        schema="rag",
    )


def downgrade() -> None:
    op.drop_index("uq_agent_steps_run_id_number", table_name="agent_steps", schema="rag")
    op.drop_table("agent_steps", schema="rag")
    op.drop_index("ix_agent_runs_collection_id_created_at", table_name="agent_runs", schema="rag")
    op.drop_table("agent_runs", schema="rag")
