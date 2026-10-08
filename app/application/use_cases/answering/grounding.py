"""A second chance for an answer that cites no sources.

Models sometimes write a correct, source-based answer and still leave out the
[n] markers: gpt-oss quoted a passage word for word under "Examples from the
text:" without a single [1]. The grounding guard (see citations.is_grounded)
can't tell that apart from an answer written from memory, so it would withhold it.

Before withholding, the use cases ask the model once to add its citations, in
the same conversation (so it sees the same sources and its own draft). The guard
itself stays strict: the rewritten answer is checked exactly like the draft, and
withheld if it still cites nothing.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.application.ports.chat import (
    ChatMessage,
    ChatModel,
    CompletionDone,
    TextDelta,
    TokenUsage,
)
from app.application.prompts.answer import CITATION_REPAIR_PROMPT
from app.application.use_cases.answering.citations import normalize_citation_marks


@dataclass(frozen=True, slots=True)
class CitationRepair:
    answer: str
    usage: TokenUsage | None


async def add_missing_citations(
    chat: ChatModel, messages: Sequence[ChatMessage], draft: str
) -> CitationRepair:
    """Ask the model to rewrite `draft` with citations. No tools are offered."""
    request = [
        *messages,
        ChatMessage("assistant", draft),
        ChatMessage("user", CITATION_REPAIR_PROMPT),
    ]
    parts: list[str] = []
    usage: TokenUsage | None = None
    async for event in chat.stream(request):
        match event:
            case TextDelta(text=raw):
                parts.append(normalize_citation_marks(raw))
            case CompletionDone(usage=final_usage):
                usage = final_usage
    return CitationRepair("".join(parts).strip(), usage)


def answer_outcome(cited: Sequence[int], *, revised: bool, withheld: bool) -> str:
    """How an answer ended, for metrics: cited, revised (cited after the retry),
    declined ("couldn't find"), or withheld."""
    if withheld:
        return "withheld"
    if revised:
        return "revised"
    return "cited" if cited else "declined"
