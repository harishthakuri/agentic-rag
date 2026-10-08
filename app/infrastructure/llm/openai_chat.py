"""Chat completions through any OpenAI-compatible endpoint (Ollama, OpenAI, vLLM).

Reasoning models (gpt-oss, OpenAI o-series) "think" before answering. Ollama
streams that thinking in a separate `reasoning` field; we only forward the
answer text (`content`), so reasoning never leaks into the answer.

Each call is a "generation" span (OpenTelemetry GenAI conventions): model, token
usage, time to first token, and with content capture the messages and the answer.
"""

import time
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
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
from app.application.ports.telemetry import (
    LLM_DURATION,
    LLM_TOKENS,
    NOOP_TELEMETRY,
    Span,
    Telemetry,
)


class OpenAICompatibleChatModel:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        *,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        telemetry: Telemetry = NOOP_TELEMETRY,
        provider: str = "openai_compatible",
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._telemetry = telemetry
        self._provider = provider

    @property
    def model(self) -> str:
        return self._model

    def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[ChatStreamEvent]:
        # The span starts here, not inside the generator: its parent is the span that
        # is current when the caller asks for the stream.
        span = self._telemetry.start_span(
            f"chat {self._model}",
            kind="generation",
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": self._provider,
                "gen_ai.request.model": self._model,
                "gen_ai.request.temperature": self._temperature,
                "gen_ai.request.reasoning_effort": self._reasoning_effort,
                "rag.tools_offered": len(tools),
            },
        )
        span.content(input=[_message_content(m) for m in messages])
        return self._stream(span, messages, tools)

    async def _stream(
        self, span: Span, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> AsyncIterator[ChatStreamEvent]:
        started = time.perf_counter()
        first_token: float | None = None
        answer: list[str] = []
        usage: TokenUsage | None = None
        calls: dict[int, _PartialToolCall] = {}
        labels = {"gen_ai.operation.name": "chat", "gen_ai.request.model": self._model}
        try:
            async for event in self._generate(messages, tools, calls):
                if isinstance(event, TextDelta):
                    if first_token is None:
                        first_token = time.perf_counter() - started
                        # Langfuse derives "time to first token" from this timestamp.
                        span.set(
                            {
                                "langfuse.observation.completion_start_time": datetime.now(
                                    UTC
                                ).isoformat()
                            }
                        )
                    answer.append(event.text)
                    yield event
                else:
                    usage = event
            tool_calls = tuple(calls[i].finish(i) for i in sorted(calls))
            span.set(
                {
                    "gen_ai.response.model": self._model,
                    "gen_ai.usage.input_tokens": usage.prompt_tokens if usage else None,
                    "gen_ai.usage.output_tokens": usage.completion_tokens if usage else None,
                    "gen_ai.response.time_to_first_token": first_token,
                    "rag.tool_calls": [c.name for c in tool_calls] or None,
                }
            )
            span.content(
                output="".join(answer)
                or [{"tool": c.name, "arguments": c.arguments} for c in tool_calls]
            )
            if usage:
                for kind, tokens in (
                    ("input", usage.prompt_tokens),
                    ("output", usage.completion_tokens),
                ):
                    self._telemetry.record(
                        LLM_TOKENS, tokens, {**labels, "gen_ai.token.type": kind}
                    )
            yield CompletionDone(usage=usage, tool_calls=tool_calls)
        except Exception as exc:
            span.fail(exc)
            raise
        finally:
            self._telemetry.record(LLM_DURATION, time.perf_counter() - started, labels)
            span.end()

    async def _generate(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
        calls: dict[int, "_PartialToolCall"],
    ) -> AsyncIterator[TextDelta | TokenUsage]:
        """Text deltas as they arrive, then the token usage (if the server reports it)."""
        usage: TokenUsage | None = None
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
        if usage:
            yield usage


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


def _message_content(message: ChatMessage) -> dict[str, Any]:
    content: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        content["tool_calls"] = [
            {"name": c.name, "arguments": c.arguments} for c in message.tool_calls
        ]
    return content


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
