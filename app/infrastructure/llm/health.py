from openai import AsyncOpenAI


class ModelHealthCheck:
    """Readiness: the OpenAI-compatible endpoint is reachable and serves the model."""

    def __init__(self, name: str, client: AsyncOpenAI, model: str) -> None:
        self.name = name
        self._client = client
        self._model = model

    async def check(self) -> None:
        await self._client.models.retrieve(self._model)
