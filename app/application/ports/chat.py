"""Chat model port: generate text from a conversation, all at once or streamed,
optionally letting the model call tools.

Implemented over any OpenAI-compatible endpoint (Ollama locally, OpenAI, vLLM).

Tool calling, in one picture:

    → messages + tool definitions
    ← assistant: "call search_knowledge_base(query='HNSW vs IVFFlat')"   (no text)
    → we run the tool, append its result as a `tool` message, call the model again
    ← assistant: final answer text (no tool calls)
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, exactly as produced by the model (may be invalid)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema of the arguments


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()  # assistant messages that requested tools
    tool_call_id: str | None = None  # tool messages: which call this answers


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A piece of the answer, as it is generated."""

    text: str


@dataclass(frozen=True, slots=True)
class CompletionDone:
    usage: TokenUsage | None
    tool_calls: tuple[ToolCall, ...] = field(default=())  # set when the model wants tools


ChatStreamEvent = TextDelta | CompletionDone


class ChatModel(Protocol):
    @property
    def model(self) -> str: ...

    def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[ChatStreamEvent]:
        """Yield TextDelta events, then exactly one CompletionDone (carrying any tool
        calls). Raises ChatModelError if the model can't be reached or fails."""
        ...


class ChatModelError(Exception):
    """The chat model is unavailable or failed. Retrying later may help."""
