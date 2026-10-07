from dataclasses import dataclass
from typing import Protocol

from app.application.ports.parsing import ParsedDocument


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """A chunk before it has an embedding (and therefore before it is a domain Chunk)."""

    text: str
    contextual_text: str
    heading_path: tuple[str, ...]
    page: int | None
    token_count: int


class Chunker(Protocol):
    def chunk(self, document: ParsedDocument, *, title: str) -> list[ChunkDraft]: ...
