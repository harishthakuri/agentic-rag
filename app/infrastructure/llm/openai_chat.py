"""Chat completions through any OpenAI-compatible endpoint (Ollama, OpenAI, vLLM).

Reasoning models (gpt-oss, OpenAI o-series) "think" before answering. Ollama
streams that thinking in a separate `reasoning` field; we only forward the
answer text (`content`), so reasoning never leaks into the answer.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

import openai
from openai import AsyncOpenAI, omit
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from app.application.ports.chat import (
    ChatMessage,
    ChatModelError,
    ChatStreamEvent,
    CompletionDone,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolSpec,
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

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[ChatStreamEvent]:
        usage: TokenUsage | None = None
        calls: dict[int, _PartialToolCall] = {}
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=[_to_openai(m) for m in messages],
                tools=[_tool(t) for t in tools] if tools else omit,
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
                    # Tool calls may arrive whole (Ollama) or in fragments (OpenAI),
                    # keyed by index: accumulate either way.
                    for fragment in choice.delta.tool_calls or ():
                        partial = calls.setdefault(fragment.index, _PartialToolCall())
                        partial.add(fragment.id, fragment.function)
        except openai.APIError as exc:
            raise ChatModelError(f"chat model request failed: {exc}") from exc
        tool_calls = tuple(calls[i].finish(i) for i in sorted(calls))
        yield CompletionDone(usage=usage, tool_calls=tool_calls)


class _PartialToolCall:
    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.arguments = ""

    def add(self, call_id: str | None, function: Any) -> None:
        if call_id:
            self.id = call_id
        if function is not None:
            self.name += function.name or ""
            self.arguments += function.arguments or ""

    def finish(self, index: int) -> ToolCall:
        return ToolCall(id=self.id or f"call_{index}", name=self.name, arguments=self.arguments)


def _to_openai(message: ChatMessage) -> ChatCompletionMessageParam:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": message.content,
        }
    if message.role == "assistant" and message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ],
        }
    return cast("ChatCompletionMessageParam", {"role": message.role, "content": message.content})


def _tool(spec: ToolSpec) -> ChatCompletionFunctionToolParam:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }
