"""Chat model port: generate text from a conversation, all at once or streamed.

Implemented over any OpenAI-compatible endpoint (Ollama locally, OpenAI, vLLM).
Tool calling for the agent extends this port in a later phase.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A piece of the answer, as it is generated."""

    text: str


@dataclass(frozen=True, slots=True)
class CompletionDone:
    usage: TokenUsage | None


ChatStreamEvent = TextDelta | CompletionDone


class ChatModel(Protocol):
    @property
    def model(self) -> str: ...

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[ChatStreamEvent]:
        """Yield TextDelta events, then exactly one CompletionDone.
        Raises ChatModelError if the model can't be reached or fails."""
        ...


class ChatModelError(Exception):
    """The chat model is unavailable or failed. Retrying later may help."""
