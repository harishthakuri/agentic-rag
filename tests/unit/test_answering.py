from uuid import UUID

import pytest

from app.application.ports.search import ChunkMatch
from app.application.prompts import answer as prompts
from app.application.use_cases.answering import (
    AnswerCompleted,
    AnswerDelta,
    AskQuery,
    AskQuestion,
    ContextBuilder,
    SourcesFound,
)
from app.application.use_cases.answering.citations import (
    check_citations,
    normalize_citation_marks,
)
from app.application.use_cases.retrieval import SearchCollection, SearchHit
from app.domain.value_objects import new_id
from tests.fakes import FakeChatModel, FakeEmbedder, FakeReranker, InMemoryStore, InMemoryUnitOfWork
from tests.unit.test_chunker import WordCounter
from tests.unit.test_retrieval import SPEC, seed_collection


# --- Citations ----------------------------------------------------------------
@pytest.mark.parametrize(
    ("answer", "cited", "invalid"),
    [
        ("Pods get an IP [1].", [1], []),
        ("Both [2][1] and again [1].", [2, 1], []),
        ("Grouped [1, 3].", [1, 3], []),
        ("Made up [7].", [], [7]),
        ("A link [1](http://x) and code `a[0]` are not citations.", [], []),
        ("No citations at all.", [], []),
    ],
)
def test_check_citations(answer: str, cited: list[int], invalid: list[int]) -> None:
    check = check_citations(answer, source_count=3)
    assert (check.cited, check.invalid) == (cited, invalid)


def test_lenticular_brackets_are_normalised_even_when_split() -> None:
    pieces = ["copy.【", "1", "】【2】"]
    assert "".join(normalize_citation_marks(p) for p in pieces) == "copy.[1][2]"


# --- Context assembly ---------------------------------------------------------
DOC_A, DOC_B = new_id(), new_id()


def _hit(doc: UUID, ordinal: int, text: str, rank: int, rerank: float | None = None) -> SearchHit:
    match = ChunkMatch(
        chunk_id=new_id(),
        document_id=doc,
        document_title="Doc A" if doc == DOC_A else "Doc B",
        ordinal=ordinal,
        text=text,
        heading_path=("Section",),
        page=None,
        score=1.0,
    )
    return SearchHit(match=match, score=1.0, retrieval_rank=rank, rerank_score=rerank)


def test_context_drops_chunks_graded_irrelevant() -> None:
    hits = [_hit(DOC_A, 0, "useful", 1, rerank=3), _hit(DOC_B, 0, "noise", 2, rerank=0)]
    sources = ContextBuilder(WordCounter(), min_rerank_score=1).build(hits)
    assert [s.text for s in sources] == ["useful"]


def test_context_merges_adjacent_chunks_and_removes_overlap() -> None:
    hits = [
        _hit(DOC_A, 4, "p2\n\np3", 1),  # most relevant, comes first
        _hit(DOC_B, 0, "other doc", 2),
        _hit(DOC_A, 3, "p1\n\np2", 3),  # neighbour of ordinal 4; "p2" is the overlap
    ]

    sources = ContextBuilder(WordCounter()).build(hits)

    assert [(s.number, s.text) for s in sources] == [(1, "p1\n\np2\n\np3"), (2, "other doc")]
    assert len(sources[0].chunk_ids) == 2


def test_context_respects_token_budget_but_fills_it() -> None:
    hits = [
        _hit(DOC_A, 0, "one two three", 1),
        _hit(DOC_A, 5, "a b c d e f g h", 2),  # too big for what's left: skipped
        _hit(DOC_B, 0, "four five", 3),  # still fits
    ]
    sources = ContextBuilder(WordCounter(), token_budget=6).build(hits)
    assert [s.text for s in sources] == ["one two three", "four five"]
    assert [s.number for s in sources] == [1, 2]


# --- AskQuestion --------------------------------------------------------------
@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


def _ask(
    store: InMemoryStore, chat: FakeChatModel, reranker: FakeReranker | None = None
) -> AskQuestion:
    search = SearchCollection(lambda: InMemoryUnitOfWork(store), FakeEmbedder(SPEC), reranker)
    return AskQuestion(search, chat, ContextBuilder(WordCounter()))


@pytest.fixture
async def collection_id(store: InMemoryStore) -> UUID:
    return (await seed_collection(store, FakeEmbedder(SPEC))).id


async def test_ask_streams_sources_then_tokens_then_done(
    store: InMemoryStore, collection_id: UUID
) -> None:
    chat = FakeChatModel("Ingress routes HTTP traffic 【1】.")
    events = [
        e async for e in _ask(store, chat).stream(AskQuery(collection_id, "ingress routes http"))
    ]

    assert isinstance(events[0], SourcesFound)
    assert all(isinstance(e, AnswerDelta) for e in events[1:-1])
    done = events[-1]
    assert isinstance(done, AnswerCompleted)
    assert done.answer == "Ingress routes HTTP traffic [1]."  # normalised
    assert done.cited == [1] and done.invalid_citations == []
    assert done.model == "fake-chat"
    assert {"search", "first_token", "generate", "total"} <= done.timings_ms.keys()


async def test_prompt_contains_delimited_sources_and_question(
    store: InMemoryStore, collection_id: UUID
) -> None:
    chat = FakeChatModel("ok [1]")
    await _ask(store, chat).execute(AskQuery(collection_id, "ingress routes http"))

    system, user = chat.calls[0]
    assert system.content == prompts.SYSTEM_PROMPT
    assert '<source id="1" location="Guide">' in user.content
    assert user.content.rstrip().endswith("Question: ingress routes http")


async def test_invalid_citations_are_reported(store: InMemoryStore, collection_id: UUID) -> None:
    result = await _ask(store, FakeChatModel("Something [9].")).execute(
        AskQuery(collection_id, "ingress")
    )
    assert result.invalid_citations == [9]


async def test_no_relevant_sources_means_no_llm_call(
    store: InMemoryStore, collection_id: UUID
) -> None:
    chat = FakeChatModel("should not be used")
    result = await _ask(store, chat, FakeReranker()).execute(  # FakeReranker grades all 0
        AskQuery(collection_id, "ingress routes http")
    )

    assert chat.calls == []
    assert result.sources == []
    assert result.answer == prompts.NO_SOURCES_ANSWER
    assert result.model is None


def test_citations_attached_to_words_count_but_code_does_not() -> None:
    answer = "Use ipvs mode[1]. In code, `items[0]` and\n```\nrows[2]\n```\nare not citations."
    check = check_citations(answer, source_count=3)
    assert (check.cited, check.invalid) == ([1], [])


def test_context_does_not_merge_neighbours_from_different_sections() -> None:
    def hit(ordinal: int, section: str, rank: int) -> SearchHit:
        match = ChunkMatch(
            chunk_id=new_id(),
            document_id=DOC_A,
            document_title="Doc A",
            ordinal=ordinal,
            text=f"text of {section}",
            heading_path=(section,),
            page=None,
            score=1.0,
        )
        return SearchHit(match=match, score=1.0, retrieval_rank=rank)

    sources = ContextBuilder(WordCounter()).build([hit(6, "Vary", 1), hit(7, "Cache busting", 2)])

    assert [s.heading_path for s in sources] == [("Vary",), ("Cache busting",)]
