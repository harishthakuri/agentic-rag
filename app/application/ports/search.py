"""Search port: the two first-stage retrievers, scoped to one collection.

Both return candidates *best first*. Fusion and reranking happen above this
port, in the application layer, so they stay storage-independent.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True, slots=True)
class ChunkMatch:
    chunk_id: UUID
    document_id: UUID
    document_title: str
    ordinal: int
    text: str
    heading_path: tuple[str, ...]
    page: int | None
    score: float  # vector: cosine similarity (higher = closer); keyword: ts_rank_cd


class ChunkSearchIndex(Protocol):
    async def vector_search(
        self, collection_id: UUID, embedding: Sequence[float], limit: int
    ) -> list[ChunkMatch]:
        """Nearest chunks by cosine distance (semantic similarity)."""
        ...

    async def keyword_search(self, collection_id: UUID, query: str, limit: int) -> list[ChunkMatch]:
        """Chunks sharing (stemmed) words with the query, ranked by full-text relevance."""
        ...

    async def neighbours(
        self, collection_id: UUID, chunk_id: UUID, before: int, after: int
    ) -> list[ChunkMatch]:
        """The chunk and up to `before`/`after` adjacent chunks of the same document,
        in document order (score 0). Empty if the chunk is not in the collection."""
        ...
