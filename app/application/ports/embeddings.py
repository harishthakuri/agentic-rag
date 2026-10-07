from collections.abc import Sequence
from typing import Protocol

from app.domain.value_objects import EmbeddingSpec


class EmbeddingProvider(Protocol):
    """Turns text into vectors.

    Documents and queries are embedded through separate methods because many
    modern embedding models (Qwen3-Embedding, E5, BGE, ...) are *asymmetric*:
    queries carry a task instruction, documents don't. Mixing them up silently
    lowers retrieval quality.
    """

    @property
    def spec(self) -> EmbeddingSpec: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class EmbeddingUnavailableError(Exception):
    """The embedding service failed (timeout, connection, 5xx). Retrying may help."""
