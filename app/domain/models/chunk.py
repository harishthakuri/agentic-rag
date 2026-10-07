from dataclasses import dataclass, field
from uuid import UUID

from app.domain.exceptions import DomainValidationError
from app.domain.models.entity import Entity
from app.domain.value_objects import new_id


@dataclass(eq=False, kw_only=True, slots=True)
class Chunk(Entity):
    """A retrievable passage of a document, with its embedding.

    `text` is what users and the LLM read. `contextual_text` is what gets
    embedded and keyword-indexed: the same text prefixed with the document
    title and heading path, so a passage like "it defaults to port 80" still
    carries what "it" refers to.
    """

    id: UUID = field(default_factory=new_id)
    document_id: UUID
    collection_id: UUID
    ordinal: int  # position within the document, 0-based
    text: str
    contextual_text: str
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    token_count: int
    embedding: list[float]

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise DomainValidationError("Chunk text must not be empty")
        if self.ordinal < 0:
            raise DomainValidationError("Chunk ordinal must not be negative")
        if not self.embedding:
            raise DomainValidationError("Chunk must have an embedding")
