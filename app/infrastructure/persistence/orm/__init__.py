"""ORM entities. Import every entity here so Alembic autogenerate can see it."""

from app.infrastructure.persistence.orm.api_key_entity import ApiKeyEntity
from app.infrastructure.persistence.orm.base import Base
from app.infrastructure.persistence.orm.chunk_entity import EMBEDDING_DIMENSIONS, ChunkEntity
from app.infrastructure.persistence.orm.collection_entity import CollectionEntity
from app.infrastructure.persistence.orm.document_entity import DocumentEntity
from app.infrastructure.persistence.orm.ingestion_job_entity import IngestionJobEntity

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "ApiKeyEntity",
    "Base",
    "ChunkEntity",
    "CollectionEntity",
    "DocumentEntity",
    "IngestionJobEntity",
]
