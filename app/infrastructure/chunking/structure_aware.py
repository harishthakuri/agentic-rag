"""Structure-aware chunking.

Why not "every 500 characters"? Fixed-size splitting cuts sentences, code
blocks and tables in half, and mixes unrelated sections in one chunk. Each
chunk's embedding then averages several topics and matches none of them well.

This chunker respects the document's own structure, from coarse to fine:

1. **Sections** (from the parser: one per heading / PDF page) are never mixed.
2. A section that fits the token budget becomes one chunk, as is.
3. Larger sections are split into **paragraphs**; fenced code blocks stay whole.
4. A paragraph that alone exceeds the budget is split into **sentences**, and as
   a last resort into fixed token windows.
5. Pieces are packed greedily up to `target_tokens`. Consecutive chunks of the
   same section share an **overlap** of whole trailing paragraphs/sentences
   (up to `overlap_tokens`), so a fact spanning a boundary appears complete in
   at least one chunk.

Each chunk also gets a **contextual header** (document title + heading path)
that is embedded with it; see `Chunk.contextual_text`.
"""

import re

from app.application.ports.chunking import ChunkDraft
from app.application.ports.parsing import ParsedDocument, Section
from app.infrastructure.chunking.tokens import TokenCounter

_BLANK_LINES = re.compile(r"\n[ \t]*\n")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})", re.MULTILINE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


class StructureAwareChunker:
    def __init__(
        self, tokens: TokenCounter, *, target_tokens: int = 500, overlap_tokens: int = 60
    ) -> None:
        if not 0 <= overlap_tokens < target_tokens:
            raise ValueError("overlap_tokens must be between 0 and target_tokens")
        self._tokens = tokens
        self._target = target_tokens
        self._overlap = overlap_tokens
        # Code is less useful when split, so a code block may use twice the budget.
        self._max_code = target_tokens * 2

    def chunk(self, document: ParsedDocument, *, title: str) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for section in document.sections:
            for text in self._split_section(section.text):
                drafts.append(self._draft(text, section, title))
        return drafts

    # --- Splitting ---------------------------------------------------------
    def _split_section(self, text: str) -> list[str]:
        if self._tokens.count(text) <= self._target:
            return [text]
        units: list[str] = []
        for block in _blocks(text):
            size = self._tokens.count(block)
            if size <= self._target or (_is_code(block) and size <= self._max_code):
                units.append(block)
            elif _is_code(block):
                units.extend(self._pack(block.split("\n"), "\n"))
            else:
                units.extend(self._pack(self._sentences(block), " "))
        return self._pack(units, "\n\n")

    def _sentences(self, paragraph: str) -> list[str]:
        pieces: list[str] = []
        for sentence in _SENTENCE_END.split(paragraph):
            if self._tokens.count(sentence) <= self._target:
                pieces.append(sentence)
            else:  # e.g. a giant table row or minified text
                pieces.extend(self._tokens.split(sentence, self._target))
        return pieces

    def _pack(self, pieces: list[str], separator: str) -> list[str]:
        """Greedily group pieces up to the token budget, with whole-piece overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for piece in pieces:
            size = self._tokens.count(piece)
            if current and current_tokens + size > self._target:
                chunks.append(separator.join(current))
                current, current_tokens = self._overlap_tail(current)
            current.append(piece)
            current_tokens += size
        if current:
            chunks.append(separator.join(current))
        return chunks

    def _overlap_tail(self, pieces: list[str]) -> tuple[list[str], int]:
        """The trailing pieces that fit in the overlap budget, to start the next chunk."""
        tail: list[str] = []
        total = 0
        for piece in reversed(pieces):
            size = self._tokens.count(piece)
            if total + size > self._overlap:
                break
            tail.insert(0, piece)
            total += size
        return tail, total

    # --- Output ------------------------------------------------------------
    def _draft(self, text: str, section: Section, title: str) -> ChunkDraft:
        header = [f"Document: {title}"]
        if section.heading_path:
            header.append(f"Section: {' > '.join(section.heading_path)}")
        return ChunkDraft(
            text=text,
            contextual_text="\n".join(header) + "\n\n" + text,
            heading_path=section.heading_path,
            page=section.page,
            token_count=self._tokens.count(text),
        )


def _blocks(text: str) -> list[str]:
    """Paragraphs separated by blank lines, but a fenced code block is one block
    even when it contains blank lines."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for paragraph in _BLANK_LINES.split(text):
        current.append(paragraph)
        # An odd number of fence markers toggles whether we're inside a code block.
        if len(_FENCE.findall(paragraph)) % 2 == 1:
            in_fence = not in_fence
        if not in_fence:
            blocks.append("\n\n".join(current).strip())
            current = []
    if current:  # unclosed fence: keep what we have
        blocks.append("\n\n".join(current).strip())
    return [b for b in blocks if b]


def _is_code(block: str) -> bool:
    return bool(_FENCE.match(block))
