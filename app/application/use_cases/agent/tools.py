"""The agent's tools. All read-only and scoped to one collection.

Every passage the agent sees gets a run-wide source number ([1], [2], ...),
assigned the first time it appears and reused afterwards, so the agent can
cite consistently across several searches.
"""

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.application.ports.chat import ToolCall, ToolSpec
from app.application.ports.search import ChunkMatch
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.use_cases.retrieval import SearchCollection, SearchMode, SearchQuery

SEARCH = ToolSpec(
    name="search_knowledge_base",
    description=(
        "Search the document collection (semantic + keyword). Returns the most relevant "
        "passages, each with a source number to cite. Use one topic per query: for a "
        "question with several parts, call this once per part."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "One topic, a few specific words, e.g. 'IVFFlat lists probes'",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)

READ_MORE_CONTEXT = ToolSpec(
    name="read_more_context",
    description=(
        "Read the passages immediately before and after a source, from the same "
        "document. Use when a passage seems cut off or lacks context."
    ),
    parameters={
        "type": "object",
        "properties": {
            "source": {"type": "integer", "description": "A source number from earlier results"}
        },
        "required": ["source"],
        "additionalProperties": False,
    },
)

LIST_DOCUMENTS = ToolSpec(
    name="list_documents",
    description="List the titles of the documents in the collection.",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)

TOOLS = (SEARCH, READ_MORE_CONTEXT, LIST_DOCUMENTS)

# Repeated where the model is looking: a rule stated only in the system prompt,
# several messages back, was ignored in practice (answers came back uncited).
CITE_REMINDER = "\n\n(When you answer, cite each fact with its source number, like [2].)"


@dataclass(frozen=True, slots=True)
class Source:
    number: int
    match: ChunkMatch

    @property
    def location(self) -> str:
        parts = [self.match.document_title, *self.match.heading_path]
        location = " > ".join(parts)
        return f"{location} (p. {self.match.page})" if self.match.page else location


class SourceRegistry:
    def __init__(self) -> None:
        self._by_number: dict[int, Source] = {}
        self._by_chunk: dict[UUID, Source] = {}

    def register(self, match: ChunkMatch) -> Source:
        existing = self._by_chunk.get(match.chunk_id)
        if existing:
            return existing
        source = Source(number=len(self._by_number) + 1, match=match)
        self._by_number[source.number] = source
        self._by_chunk[match.chunk_id] = source
        return source

    def get(self, number: int) -> Source | None:
        return self._by_number.get(number)

    def all(self) -> list[Source]:
        return list(self._by_number.values())

    def __len__(self) -> int:
        return len(self._by_number)


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    content: str  # what the model reads
    arguments: dict[str, Any]  # parsed arguments (for events and the run log)
    summary: dict[str, Any] = field(default_factory=dict)  # what clients and the run log see


class AgentToolbox:
    def __init__(
        self,
        collection_id: UUID,
        search: SearchCollection,
        uow_factory: UnitOfWorkFactory,
        registry: SourceRegistry,
        *,
        top_k: int = 5,
        rerank: bool = False,
        max_passage_chars: int = 1200,
    ) -> None:
        self._collection_id = collection_id
        self._search = search
        self._uow_factory = uow_factory
        self._registry = registry
        self._top_k = top_k
        self._rerank = rerank
        self._max_chars = max_passage_chars
        self._queries: set[str] = set()
        self.listed_documents = False  # list_documents was called at least once

    async def run(self, call: ToolCall) -> ToolOutcome:
        """Never raises for bad input: errors are reported to the model as text,
        so it can correct itself (wrong argument, unknown tool, repeated query)."""
        arguments = parse_arguments(call)
        if "raw" in arguments:
            return _error(arguments, "Invalid arguments. Send a JSON object.")

        match call.name:
            case SEARCH.name:
                return await self._search_tool(arguments)
            case READ_MORE_CONTEXT.name:
                return await self._read_more(arguments)
            case LIST_DOCUMENTS.name:
                return await self._list_documents(arguments)
            case _:
                names = ", ".join(t.name for t in TOOLS)
                return _error(arguments, f"Unknown tool '{call.name}'. Available: {names}.")

    async def _search_tool(self, arguments: dict[str, Any]) -> ToolOutcome:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return _error(arguments, "The 'query' argument is required.")
        normalized = " ".join(query.lower().split())
        if normalized in self._queries:
            return _error(
                arguments,
                "You already ran this exact search. Try different words, or answer with "
                "the sources you have.",
            )
        self._queries.add(normalized)

        result = await self._search.execute(
            SearchQuery(
                self._collection_id,
                query,
                mode=SearchMode.HYBRID,
                top_k=self._top_k,
                rerank=self._rerank,
            )
        )
        sources = [self._registry.register(hit.match) for hit in result.hits]
        if not sources:
            return ToolOutcome(f"No results for '{query}'.", arguments, {"sources": []})
        return ToolOutcome(
            f"Results for '{query}':\n\n" + self._format(sources) + CITE_REMINDER,
            arguments,
            _summary(sources),
        )

    async def _read_more(self, arguments: dict[str, Any]) -> ToolOutcome:
        number = arguments.get("source")
        source = self._registry.get(number) if isinstance(number, int) else None
        if source is None:
            return _error(
                arguments, f"Unknown source {number!r}. Use a number from earlier results."
            )
        async with self._uow_factory() as uow:
            matches = await uow.search.neighbours(
                self._collection_id, source.match.chunk_id, before=1, after=1
            )
        sources = [self._registry.register(m) for m in matches]
        return ToolOutcome(
            f"Source [{source.number}] with its neighbouring passages, in document order:\n\n"
            + self._format(sources)
            + CITE_REMINDER,
            arguments,
            _summary(sources),
        )

    async def _list_documents(self, arguments: dict[str, Any]) -> ToolOutcome:
        self.listed_documents = True
        async with self._uow_factory() as uow:
            documents = await uow.documents.list_by_collection(
                self._collection_id, limit=100, offset=0
            )
        titles = [d.title for d in documents]
        listing = "\n".join(f"- {title}" for title in titles) or "(the collection is empty)"
        return ToolOutcome(f"{len(titles)} documents:\n{listing}", arguments, {"documents": titles})

    def _format(self, sources: list[Source]) -> str:
        return "\n\n".join(
            f"[{s.number}] {s.location}\n{_truncate(s.match.text, self._max_chars)}"
            for s in sources
        )


def parse_arguments(call: ToolCall) -> dict[str, Any]:
    """The call's arguments as a dict, or {"raw": text} if they aren't a JSON object."""
    try:
        arguments = json.loads(call.arguments or "{}")
    except ValueError:
        return {"raw": call.arguments}
    return arguments if isinstance(arguments, dict) else {"raw": call.arguments}


def _summary(sources: list[Source]) -> dict[str, Any]:
    return {"sources": [{"number": s.number, "location": s.location} for s in sources]}


def _error(arguments: dict[str, Any], message: str) -> ToolOutcome:
    return ToolOutcome(f"Error: {message}", arguments, {"error": message})


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"
