from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from app.domain.models.entity import Entity
from app.domain.value_objects import CollectionName, EmbeddingSpec, new_id


@dataclass(eq=False, kw_only=True, slots=True)
class Collection(Entity):
    """A named set of documents searched together, embedded with one model."""

    id: UUID = field(default_factory=new_id)
    name: CollectionName
    description: str | None = None
    embedding: EmbeddingSpec
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls, *, name: CollectionName, embedding: EmbeddingSpec, description: str | None = None
    ) -> "Collection":
        cleaned = description.strip() if description else None
        return cls(name=name, embedding=embedding, description=cleaned or None)
