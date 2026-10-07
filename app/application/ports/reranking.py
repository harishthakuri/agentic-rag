"""Reranking: a slower, more precise second stage over first-stage candidates.

First-stage retrievers (vector, keyword) judge each chunk *independently of the
query's details*: a vector compares two precomputed embeddings, full-text search
counts matching words. A reranker reads the query and each candidate *together*
and judges whether the passage actually answers the question. Too expensive for
the whole corpus, affordable for the top 20 or so candidates.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    id: UUID
    text: str  # what the reranker reads: title, heading path and passage


@dataclass(frozen=True, slots=True)
class RerankScore:
    id: UUID
    score: float  # higher = more relevant; the scale is reranker-specific


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    async def rerank(self, query: str, candidates: Sequence[RerankCandidate]) -> list[RerankScore]:
        """Score every candidate (any order). Raises RerankError on failure."""
        ...


class RerankError(Exception):
    """The reranker failed (unreachable, timed out, unusable output). Search then
    falls back to the first-stage order instead of failing the request."""
