"""Embeddings through any OpenAI-compatible endpoint (Ollama locally, or OpenAI).

`dimensions` asks the server to truncate vectors. Qwen3-Embedding and OpenAI's
text-embedding-3 models are trained with Matryoshka Representation Learning,
so their first N dimensions carry most of the meaning. 4096 → 1024 keeps
quality close to the full vector while fitting pgvector's HNSW index.
"""

import math
from collections.abc import Sequence

import openai
from openai import AsyncOpenAI

from app.application.ports.embeddings import EmbeddingUnavailableError
from app.domain.value_objects import EmbeddingSpec

# Qwen3-Embedding is instruction-aware for queries (documents get no prefix).
DEFAULT_QUERY_INSTRUCTION = (
    "Given a user question, retrieve passages from the knowledge base that answer it"
)


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        client: AsyncOpenAI,
        spec: EmbeddingSpec,
        *,
        batch_size: int = 32,
        query_instruction: str | None = DEFAULT_QUERY_INSTRUCTION,
    ) -> None:
        self._client = client
        self._spec = spec
        self._batch_size = batch_size
        self._query_instruction = query_instruction

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(await self._embed(list(texts[start : start + self._batch_size])))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        if self._query_instruction:
            text = f"Instruct: {self._query_instruction}\nQuery: {text}"
        [vector] = await self._embed([text])
        return vector

    async def _embed(self, batch: list[str]) -> list[list[float]]:
        try:
            response = await self._client.embeddings.create(
                model=self._spec.model, input=batch, dimensions=self._spec.dimensions
            )
        except (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        ) as exc:
            raise EmbeddingUnavailableError(f"embedding service unavailable: {exc}") from exc

        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [_normalize(item.embedding) for item in ordered]
        for vector in vectors:
            if len(vector) != self._spec.dimensions:
                raise ValueError(
                    f"{self._spec.model} returned {len(vector)} dimensions, expected "
                    f"{self._spec.dimensions}; does the endpoint support `dimensions`?"
                )
        return vectors


def _normalize(vector: list[float]) -> list[float]:
    """Unit length, so cosine similarity equals the dot product. Truncated Matryoshka
    vectors are not unit length unless the server re-normalises them."""
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector
