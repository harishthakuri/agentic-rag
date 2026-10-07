from datetime import datetime
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, ForeignKey, Index, Text, func
from sqlalchemy import text as sql_text  # `text` is also a column name below
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.orm.base import Base

# Fixed by the schema: changing it requires a migration and re-embedding.
# Settings.embedding_dim is checked against this at startup.
EMBEDDING_DIMENSIONS = 1024


class ChunkEntity(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("uq_chunks_document_id_ordinal", "document_id", "ordinal", unique=True),
        Index("ix_chunks_collection_id", "collection_id"),
        # Approximate nearest-neighbour index for cosine distance (`<=>`).
        #   m: links per node (higher = better recall, more memory)
        #   ef_construction: candidates considered while building (higher = better graph, slower)
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # Inverted index for full-text (keyword) search.
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    # Denormalised from documents so searches filter without a join.
    collection_id: Mapped[UUID] = mapped_column(ForeignKey("collections.id", ondelete="CASCADE"))
    ordinal: Mapped[int]  # position within the document
    text: Mapped[str] = mapped_column(Text)  # original text, shown to users and the LLM
    contextual_text: Mapped[str] = mapped_column(Text)  # title + headings + text; embedded
    heading_path: Mapped[list[str]] = mapped_column(
        ARRAY(Text), server_default=sql_text("'{}'::text[]")
    )
    page: Mapped[int | None]
    token_count: Mapped[int]
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    # Maintained by PostgreSQL: always in sync with contextual_text.
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', contextual_text)", persisted=True)
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, server_default=sql_text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
