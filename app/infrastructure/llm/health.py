from app.infrastructure.llm.openai_embedder import OpenAICompatibleEmbedder


class EmbeddingHealthCheck:
    name = "embedding_model"

    def __init__(self, embedder: OpenAICompatibleEmbedder) -> None:
        self._embedder = embedder

    async def check(self) -> None:
        await self._embedder.check()
