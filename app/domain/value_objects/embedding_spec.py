from dataclasses import dataclass

from app.domain.exceptions import DomainValidationError


@dataclass(frozen=True, slots=True)
class EmbeddingSpec:
    """Which embedding model (and output size) produced a collection's vectors.

    Vectors from different models live in unrelated spaces: comparing them
    gives meaningless similarity scores. Each collection therefore records
    the spec it was built with, and ingestion/search refuse a mismatch.
    """

    model: str
    dimensions: int

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise DomainValidationError("Embedding model name must not be empty")
        if self.dimensions <= 0:
            raise DomainValidationError("Embedding dimensions must be positive")
