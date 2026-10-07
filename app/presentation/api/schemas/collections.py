from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.domain.models import Collection


class CreateCollectionRequest(BaseModel):
    name: str = Field(min_length=3, max_length=64, examples=["kubernetes-docs"])
    description: str | None = Field(default=None, max_length=1000)


class CollectionResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    embedding_model: str
    embedding_dim: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, collection: Collection) -> "CollectionResponse":
        return cls(
            id=collection.id,
            name=collection.name.value,
            description=collection.description,
            embedding_model=collection.embedding.model,
            embedding_dim=collection.embedding.dimensions,
            created_at=collection.created_at,
            updated_at=collection.updated_at,
        )
