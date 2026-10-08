from collections.abc import Sequence
from uuid import UUID

import pytest

from app.application.ports.chat import ChatMessage, ChatModelError
from app.application.ports.search import ChunkMatch
from app.application.prompts import answer as prompts
from app.application.use_cases.answering import (
    AnswerCompleted,
    AnswerDelta,
    AnswerRevising,
    AskQuery,
    AskQuestion,
    ContextBuilder,
    SourcesFound,
)
from app.application.use_cases.answering.citations import (
    check_citations,
    is_grounded,
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


@pytest.mark.parametrize(
    ("answer", "grounded"),
    [
        ("Pods get an IP [1].", True),
        ("I couldn't find this in the documents.", True),
        ("I couldn\u2019t find this in the documents.", True),  # typographic apostrophe
        ("Set timeZone in the CronJob spec.", False),  # from the model's memory
        ("Made up [7].", False),  # cites only a source that doesn't exist
        ("", False),
    ],
)
def test_is_grounded(answer: str, grounded: bool) -> None:
    assert is_grounded(answer, check_citations(answer, source_count=3)) is grounded


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


async def test_uncited_answer_is_withheld(store: InMemoryStore, collection_id: UUID) -> None:
    # An answer from memory: the retry to add citations can't ground it either.
    chat = FakeChatModel("CronJobs take a timeZone field since v1.30.")
    events = [e async for e in _ask(store, chat).stream(AskQuery(collection_id, "ingress"))]

    # The draft was streamed, the retry announced, then the final event replaces it.
    assert any(isinstance(e, AnswerDelta) for e in events)
    assert isinstance(events[-2], AnswerRevising)
    done = events[-1]
    assert isinstance(done, AnswerCompleted)
    assert done.withheld and done.answer == prompts.UNGROUNDED_ANSWER
    assert not done.revised
    assert len(chat.calls) == 2  # the draft and one retry, no more


async def test_cited_answer_and_decline_are_not_withheld(
    store: InMemoryStore, collection_id: UUID
) -> None:
    for reply in ("Ingress routes HTTP [1].", "I couldn't find this in the documents."):
        chat = FakeChatModel(reply)
        result = await _ask(store, chat).execute(AskQuery(collection_id, "ingress"))
        assert not result.withheld and not result.revised and result.answer == reply
        assert len(chat.calls) == 1  # no retry needed


async def test_uncited_draft_gets_one_retry_to_add_citations(
    store: InMemoryStore, collection_id: UUID
) -> None:
    draft = "A quick buck is profit earned quickly. Examples from the text: ..."
    chat = FakeChatModel([draft, "A quick buck is profit earned quickly 【1】."])
    events = [e async for e in _ask(store, chat).stream(AskQuery(collection_id, "ingress"))]

    done = events[-1]
    assert isinstance(done, AnswerCompleted)
    assert done.revised and not done.withheld
    assert done.answer == "A quick buck is profit earned quickly [1]."  # normalised
    assert done.cited == [1]
    assert done.usage is not None and done.usage.prompt_tokens == 200  # both calls
    assert "repair" in done.timings_ms
    # The retry continues the same conversation: same sources, then the draft.
    first, retry = chat.calls
    assert retry[:2] == first
    assert retry[2] == ChatMessage("assistant", draft)
    assert retry[3] == ChatMessage("user", prompts.CITATION_REPAIR_PROMPT)


async def test_retry_may_decline(store: InMemoryStore, collection_id: UUID) -> None:
    chat = FakeChatModel(["From memory.", "I couldn't find this in the documents."])
    result = await _ask(store, chat).execute(AskQuery(collection_id, "ingress"))
    assert result.revised and not result.withheld
    assert result.answer == "I couldn't find this in the documents."


async def test_failed_retry_withholds_instead_of_failing(
    store: InMemoryStore, collection_id: UUID
) -> None:
    def reply(messages: Sequence[ChatMessage]) -> str:
        if messages[-1].content == prompts.CITATION_REPAIR_PROMPT:
            raise ChatModelError("simulated outage")
        return "From memory."

    result = await _ask(store, FakeChatModel(reply)).execute(AskQuery(collection_id, "ingress"))
    assert result.withheld and result.answer == prompts.UNGROUNDED_ANSWER


async def test_invalid_citations_are_reported(store: InMemoryStore, collection_id: UUID) -> None:
    result = await _ask(store, FakeChatModel("Something [9].")).execute(
        AskQuery(collection_id, "ingress")
    )
    assert result.invalid_citations == [9]
    assert result.withheld  # it cited no real source
    assert result.answer == prompts.UNGROUNDED_ANSWER


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
