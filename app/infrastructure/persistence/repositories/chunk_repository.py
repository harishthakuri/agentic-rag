from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Chunk
from app.domain.repositories import ChunkRepository
from app.infrastructure.persistence.orm import ChunkEntity


class SqlAlchemyChunkRepository(ChunkRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_for_document(self, document_id: UUID, chunks: Sequence[Chunk]) -> None:
        await self._session.execute(
            delete(ChunkEntity).where(ChunkEntity.document_id == document_id)
        )
        if chunks:
            # One multi-row INSERT instead of an ORM flush per object.
            await self._session.execute(insert(ChunkEntity), [_to_row(c) for c in chunks])

    async def count_for_document(self, document_id: UUID) -> int:
        count = await self._session.scalar(
            select(func.count())
            .select_from(ChunkEntity)
            .where(ChunkEntity.document_id == document_id)
        )
        return count or 0


def _to_row(chunk: Chunk) -> dict[str, object]:
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "collection_id": chunk.collection_id,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "contextual_text": chunk.contextual_text,
        "heading_path": list(chunk.heading_path),
        "page": chunk.page,
        "token_count": chunk.token_count,
        "embedding": chunk.embedding,
    }
