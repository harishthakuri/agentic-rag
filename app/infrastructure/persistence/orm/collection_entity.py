from uuid import UUID

from sqlalchemy import CheckConstraint, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.orm.base import Base, TimestampMixin


class CollectionEntity(TimestampMixin, Base):
    __tablename__ = "collections"
    __table_args__ = (CheckConstraint("embedding_dim > 0", name="embedding_dim_positive"),)

    id: Mapped[UUID] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(String(200))
    embedding_dim: Mapped[int]
