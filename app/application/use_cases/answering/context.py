"""Context assembly: turn search hits into the numbered sources the LLM reads.

What the model sees matters as much as which model it is:

1. **Drop irrelevant hits.** When results were reranked, chunks graded below
   `min_rerank_score` are dropped. Irrelevant context dilutes attention and
   invites the model to "use" it anyway.
2. **Merge neighbours.** Adjacent chunks of the same *section* (a long section
   the chunker had to split) become one source again, without the paragraphs
   repeated by chunk overlap. Chunks of different sections stay separate, so
   every source carries an accurate location.
3. **Number by relevance.** [1] is the most relevant source, which helps the
   model and keeps citations stable.
4. **Respect a token budget.** Sources are added in relevance order while they
   fit; anything else is left out.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from app.application.ports.tokens import TokenCounter
from app.application.use_cases.retrieval import SearchHit


@dataclass(frozen=True, slots=True)
class ContextSource:
    number: int  # cited as [number]
    document_id: UUID
    document_title: str
    heading_path: tuple[str, ...]
    page: int | None
    chunk_ids: tuple[UUID, ...]
    text: str
    token_count: int

    @property
    def location(self) -> str:
        parts = [self.document_title, *self.heading_path]
        location = " > ".join(parts)
        return f"{location} (p. {self.page})" if self.page else location


class ContextBuilder:
    def __init__(
        self, tokens: TokenCounter, *, token_budget: int = 3000, min_rerank_score: float = 1.0
    ) -> None:
        self._tokens = tokens
        self._budget = token_budget
        self._min_rerank_score = min_rerank_score

    def build(self, hits: Sequence[SearchHit]) -> list[ContextSource]:
        relevant = [
            h for h in hits if h.rerank_score is None or h.rerank_score >= self._min_rerank_score
        ]

        sources: list[ContextSource] = []
        remaining = self._budget
        for group in _adjacent_groups(relevant):
            text = _merge_texts([h.match.text for h in group])
            size = self._tokens.count(text)
            if size > remaining:
                continue  # a smaller, less relevant source may still fit
            remaining -= size
            first = group[0].match
            sources.append(
                ContextSource(
                    number=len(sources) + 1,
                    document_id=first.document_id,
                    document_title=first.document_title,
                    heading_path=first.heading_path,
                    page=first.page,
                    chunk_ids=tuple(h.match.chunk_id for h in group),
                    text=text,
                    token_count=size,
                )
            )
        return sources


def _adjacent_groups(hits: Sequence[SearchHit]) -> list[list[SearchHit]]:
    """Group hits of the same section with consecutive ordinals. Groups keep the
    position of their most relevant hit; chunks inside a group are in document order."""
    groups: list[list[SearchHit]] = []
    for hit in hits:
        group = next((g for g in groups if _touches(g, hit)), None)
        if group is None:
            groups.append([hit])
        else:
            group.append(hit)
            group.sort(key=lambda h: h.match.ordinal)
    return groups


def _touches(group: list[SearchHit], hit: SearchHit) -> bool:
    first = group[0].match
    if (first.document_id, first.heading_path) != (hit.match.document_id, hit.match.heading_path):
        return False
    return hit.match.ordinal in (group[0].match.ordinal - 1, group[-1].match.ordinal + 1)


def _merge_texts(texts: list[str]) -> str:
    """Join consecutive chunks, dropping the leading paragraphs a chunk repeats
    from the previous one (the chunker's overlap)."""
    paragraphs: list[str] = texts[0].split("\n\n")
    for text in texts[1:]:
        following = text.split("\n\n")
        overlap = next(
            (
                n
                for n in range(min(len(paragraphs), len(following)), 0, -1)
                if paragraphs[-n:] == following[:n]
            ),
            0,
        )
        paragraphs.extend(following[overlap:])
    return "\n\n".join(paragraphs)
