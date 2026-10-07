"""Chat completions through any OpenAI-compatible endpoint (Ollama, OpenAI, vLLM).

Reasoning models (gpt-oss, OpenAI o-series) "think" before answering. Ollama
streams that thinking in a separate `reasoning` field; we only forward the
answer text (`content`), so reasoning never leaks into the answer.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

import openai
from openai import AsyncOpenAI, omit
from openai.types.chat import ChatCompletionMessageParam

from app.application.ports.chat import (
    ChatMessage,
    ChatModelError,
    ChatStreamEvent,
    CompletionDone,
    TextDelta,
    TokenUsage,
)


class OpenAICompatibleChatModel:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        *,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort

    @property
    def model(self) -> str:
        return self._model

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[ChatStreamEvent]:
        payload = cast(
            "list[ChatCompletionMessageParam]",
            [{"role": m.role, "content": m.content} for m in messages],
        )
        usage: TokenUsage | None = None
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=payload,
                temperature=self._temperature,
                stream=True,
                stream_options={"include_usage": True},
                # Reasoning models only; "low" keeps answers fast.
                reasoning_effort=cast("Any", self._reasoning_effort)
                if self._reasoning_effort
                else omit,
            )
            async for chunk in stream:
                if chunk.usage:
                    usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                    )
                for choice in chunk.choices:
                    if choice.delta.content:
                        yield TextDelta(choice.delta.content)
        except openai.APIError as exc:
            raise ChatModelError(f"chat model request failed: {exc}") from exc
        yield CompletionDone(usage=usage)
