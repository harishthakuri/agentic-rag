"""Vector and keyword search over `rag.chunks` with pgvector and PostgreSQL full-text search."""

import re
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import Row, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.ports.search import ChunkMatch

# Explicit search-engine syntax: "quoted phrase", OR, -excluded
_WEBSEARCH_SYNTAX = re.compile(r'"|\bor\b|(^|\s)-\w', re.IGNORECASE)

# Order by distance in an inner query so the HNSW index can serve it, then join.
_VECTOR_SQL = text("""
    SELECT c.id, c.document_id, d.title, c.ordinal, c.text, c.heading_path, c.page,
           1 - c.distance AS score
    FROM (
        SELECT id, document_id, ordinal, text, heading_path, page,
               embedding <=> CAST(:embedding AS vector) AS distance
        FROM rag.chunks
        WHERE collection_id = :collection_id
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    ) AS c
    JOIN rag.documents AS d ON d.id = c.document_id
    ORDER BY c.distance
""")

_KEYWORD_SQL = """
    SELECT c.id, c.document_id, d.title, c.ordinal, c.text, c.heading_path, c.page,
           ts_rank_cd(c.tsv, q.query, 32) AS score
    FROM rag.chunks AS c
    JOIN rag.documents AS d ON d.id = c.document_id
    CROSS JOIN (SELECT {tsquery} AS query) AS q
    WHERE c.collection_id = :collection_id AND c.tsv @@ q.query
    ORDER BY score DESC, c.id
    LIMIT :limit
"""

# A natural-language question rarely has ALL its words in one chunk, so AND-ing
# them (what plainto/websearch_to_tsquery do) usually matches nothing. We OR the
# stemmed, stop-word-filtered terms instead and let ranking reward chunks that
# match more of them (ts_rank_cd also rewards terms appearing close together).
_ANY_TERMS = text(
    _KEYWORD_SQL.format(
        tsquery="CAST(replace(CAST(plainto_tsquery('english', :query) AS text), ' & ', ' | ')"
        " AS tsquery)"
    )
)
_NEIGHBOURS_SQL = text("""
    SELECT c.id, c.document_id, d.title, c.ordinal, c.text, c.heading_path, c.page,
           0.0 AS score
    FROM rag.chunks AS target
    JOIN rag.chunks AS c
      ON c.document_id = target.document_id
     AND c.ordinal BETWEEN target.ordinal - :before AND target.ordinal + :after
    JOIN rag.documents AS d ON d.id = c.document_id
    WHERE target.id = :chunk_id AND target.collection_id = :collection_id
    ORDER BY c.ordinal
""")

# Users who type search syntax get exactly what they asked for.
_WEBSEARCH = text(_KEYWORD_SQL.format(tsquery="websearch_to_tsquery('english', :query)"))


class PgChunkSearchIndex:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def vector_search(
        self, collection_id: UUID, embedding: Sequence[float], limit: int
    ) -> list[ChunkMatch]:
        # Both settings are transaction-local (`is_local => true`).
        # ef_search: the HNSW candidate list must be at least as large as LIMIT.
        # iterative_scan: with the collection filter applied *after* the index
        #   scan, keep scanning until enough rows match (pgvector >= 0.8).
        await self._session.execute(
            text(
                "SELECT set_config('hnsw.ef_search', :ef_search, true), "
                "set_config('hnsw.iterative_scan', 'relaxed_order', true)"
            ),
            {"ef_search": str(max(40, limit))},
        )
        rows = await self._session.execute(
            _VECTOR_SQL,
            {
                "embedding": _vector_literal(embedding),
                "collection_id": collection_id,
                "limit": limit,
            },
        )
        # relaxed_order may return slightly out-of-order rows; the outer ORDER BY fixes that.
        return [_to_match(row) for row in rows]

    async def keyword_search(self, collection_id: UUID, query: str, limit: int) -> list[ChunkMatch]:
        statement = _WEBSEARCH if _WEBSEARCH_SYNTAX.search(query) else _ANY_TERMS
        rows = await self._session.execute(
            statement, {"query": query, "collection_id": collection_id, "limit": limit}
        )
        return [_to_match(row) for row in rows]

    async def neighbours(
        self, collection_id: UUID, chunk_id: UUID, before: int, after: int
    ) -> list[ChunkMatch]:
        rows = await self._session.execute(
            _NEIGHBOURS_SQL,
            {
                "collection_id": collection_id,
                "chunk_id": chunk_id,
                "before": before,
                "after": after,
            },
        )
        return [_to_match(row) for row in rows]


def _vector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.8g}" for x in embedding) + "]"


def _to_match(row: Row[Any]) -> ChunkMatch:
    columns = row._mapping
    return ChunkMatch(
        chunk_id=columns["id"],
        document_id=columns["document_id"],
        document_title=columns["title"],
        ordinal=columns["ordinal"],
        text=columns["text"],
        heading_path=tuple(columns["heading_path"] or ()),
        page=columns["page"],
        score=float(columns["score"]),
    )
