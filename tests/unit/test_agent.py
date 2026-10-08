import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from uuid import UUID

import pytest

from app.application.ports.chat import ChatMessage, ChatStreamEvent, ToolCall, ToolSpec
from app.application.ports.unit_of_work import UnitOfWorkFactory
from app.application.prompts import agent as prompts
from app.application.prompts.answer import UNGROUNDED_ANSWER
from app.application.use_cases.agent import (
    AgentAsk,
    AgentCompleted,
    AgentQuery,
    AgentTimeoutError,
    RunStarted,
    ToolCalled,
    ToolReturned,
)
from app.application.use_cases.agent.tools import AgentToolbox, SourceRegistry
from app.application.use_cases.answering import AnswerDelta
from app.application.use_cases.collections import CollectionNotFoundError
from app.application.use_cases.retrieval import SearchCollection
from app.domain.models import RunStatus
from app.domain.value_objects import new_id
from tests.fakes import FakeChatModel, FakeEmbedder, InMemoryStore, InMemoryUnitOfWork
from tests.unit.test_retrieval import SPEC, seed_collection


def _call(name: str, call_id: str = "c1", **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def uow(store: InMemoryStore) -> UnitOfWorkFactory:
    return lambda: InMemoryUnitOfWork(store)


@pytest.fixture
async def collection_id(store: InMemoryStore) -> UUID:
    return (await seed_collection(store, FakeEmbedder(SPEC))).id


def _search(uow: UnitOfWorkFactory) -> SearchCollection:
    return SearchCollection(uow, FakeEmbedder(SPEC))


def _agent(uow: UnitOfWorkFactory, chat: FakeChatModel, **kwargs: object) -> AgentAsk:
    return AgentAsk(uow, _search(uow), chat, **kwargs)  # type: ignore[arg-type]


# --- Toolbox ------------------------------------------------------------------
async def test_search_numbers_sources_and_reuses_numbers(
    uow: UnitOfWorkFactory, collection_id: UUID
) -> None:
    registry = SourceRegistry()
    toolbox = AgentToolbox(collection_id, _search(uow), uow, registry, top_k=2)

    first = await toolbox.run(_call("search_knowledge_base", query="ingress routes http"))
    second = await toolbox.run(_call("search_knowledge_base", query="ingress http"))

    assert first.content.startswith("Results for 'ingress routes http':\n\n[1] Guide\n")
    assert "cite each fact with its source number" in first.content
    assert [s["number"] for s in first.summary["sources"]] == [1, 2]

    # A passage seen again keeps its number; new ones continue the sequence.
    def chunk_numbers(summary: dict[str, list[dict[str, int]]]) -> dict[UUID, int]:
        return {
            registry.get(s["number"]).match.chunk_id: s["number"]  # type: ignore[union-attr]
            for s in summary["sources"]
        }

    seen_first = chunk_numbers(first.summary)
    for chunk_id, number in chunk_numbers(second.summary).items():
        expected = seen_first.get(chunk_id)
        assert number == expected if expected else number > max(seen_first.values())


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (ToolCall("c", "search_knowledge_base", "not json"), "Invalid arguments"),
        (_call("search_knowledge_base"), "'query' argument is required"),
        (_call("delete_everything"), "Unknown tool 'delete_everything'"),
        (_call("read_more_context", source=99), "Unknown source 99"),
    ],
)
async def test_bad_tool_calls_become_error_text_for_the_model(
    uow: UnitOfWorkFactory, collection_id: UUID, call: ToolCall, error: str
) -> None:
    toolbox = AgentToolbox(collection_id, _search(uow), uow, SourceRegistry())
    outcome = await toolbox.run(call)
    assert outcome.content.startswith("Error: ")
    assert error in outcome.content


async def test_repeated_search_is_refused(uow: UnitOfWorkFactory, collection_id: UUID) -> None:
    toolbox = AgentToolbox(collection_id, _search(uow), uow, SourceRegistry())
    await toolbox.run(_call("search_knowledge_base", query="Pods"))
    again = await toolbox.run(_call("search_knowledge_base", query="  pods "))
    assert "already ran this exact search" in again.content


async def test_read_more_context_returns_neighbours_in_order(
    uow: UnitOfWorkFactory, collection_id: UUID
) -> None:
    registry = SourceRegistry()
    toolbox = AgentToolbox(collection_id, _search(uow), uow, registry, top_k=1)
    found = await toolbox.run(_call("search_knowledge_base", query="ingress routes http"))
    number = found.summary["sources"][0]["number"]

    more = await toolbox.run(_call("read_more_context", source=number))

    texts = [registry.get(s["number"]).match.ordinal for s in more.summary["sources"]]  # type: ignore[union-attr]
    assert texts == sorted(texts) and len(texts) == 3  # ordinals 0, 1, 2 around ordinal 1


async def test_list_documents(uow: UnitOfWorkFactory, collection_id: UUID) -> None:
    toolbox = AgentToolbox(collection_id, _search(uow), uow, SourceRegistry())
    outcome = await toolbox.run(_call("list_documents"))
    assert outcome.summary == {"documents": ["Guide"]}


# --- The loop -----------------------------------------------------------------
async def test_agent_searches_then_answers_and_records_the_run(
    uow: UnitOfWorkFactory, store: InMemoryStore, collection_id: UUID
) -> None:
    chat = FakeChatModel(
        [[_call("search_knowledge_base", query="ingress routes http")], "Ingress routes it 【1】."]
    )

    events = [e async for e in _agent(uow, chat).stream(AgentQuery(collection_id, "routing?"))]

    kinds = [type(e) for e in events]
    assert kinds[:3] == [RunStarted, ToolCalled, ToolReturned]
    assert set(kinds[3:-1]) == {AnswerDelta}
    done = events[-1]
    assert isinstance(done, AgentCompleted)
    assert done.answer == "Ingress routes it [1]."
    assert done.cited == [1] and done.tool_calls == 1
    # The tool result went back to the model, linked to its call.
    tool_message = chat.calls[1][-1]
    assert tool_message.role == "tool" and tool_message.tool_call_id == "c1"
    # The run and its step were stored.
    run = store.agent_runs[done.run_id]
    assert run.status is RunStatus.SUCCEEDED and run.step_count == 1
    assert store.agent_steps[run.id][0].arguments == {"query": "ingress routes http"}


async def test_uncited_answer_is_withheld_and_not_stored(
    store: InMemoryStore, uow: UnitOfWorkFactory, collection_id: UUID
) -> None:
    # The searches found passages, but the answer cites none of them.
    chat = FakeChatModel(
        [[_call("search_knowledge_base", query="cronjob time zone")], "Use timeZone (v1.30+)."]
    )
    done = (await _agent(uow, chat).execute(AgentQuery(collection_id, "time zone?")))[0]

    assert done.withheld and done.answer == UNGROUNDED_ANSWER
    assert store.agent_runs[done.run_id].answer == UNGROUNDED_ANSWER


async def test_decline_and_collection_overview_are_not_withheld(
    uow: UnitOfWorkFactory, collection_id: UUID
) -> None:
    decline = FakeChatModel(
        [
            [_call("search_knowledge_base", query="cronjob")],
            "I couldn't find this in the documents.",
        ]
    )
    overview = FakeChatModel([[_call("list_documents")], "The collection has one guide."])

    for chat in (decline, overview):
        done = (await _agent(uow, chat).execute(AgentQuery(collection_id, "?")))[0]
        assert not done.withheld


async def test_tool_limit_withdraws_tools_and_forces_an_answer(
    uow: UnitOfWorkFactory, collection_id: UUID
) -> None:
    def keep_searching(messages: Sequence[ChatMessage]) -> str | list[ToolCall]:
        if messages[-1].content == prompts.FINAL_ANSWER_NUDGE:
            return "Best effort answer [1]."
        return [_call("search_knowledge_base", f"c{len(messages)}", query=f"q{len(messages)}")]

    chat = FakeChatModel(keep_searching)
    done = (await _agent(uow, chat).execute(AgentQuery(collection_id, "?", max_tool_calls=2)))[0]

    assert done.tool_calls == 2
    assert done.answer == "Best effort answer [1]."
    assert chat.tools_offered[-1] == []  # the final turn had no tools
    assert len(chat.tools_offered[0]) == 3


async def test_calls_beyond_the_limit_in_one_turn_are_refused(
    uow: UnitOfWorkFactory, collection_id: UUID
) -> None:
    burst = [_call("search_knowledge_base", f"c{i}", query=f"q{i}") for i in range(3)]
    chat = FakeChatModel([burst, "done"])

    done = (await _agent(uow, chat).execute(AgentQuery(collection_id, "?", max_tool_calls=2)))[0]

    assert done.tool_calls == 2
    refused = [m for m in chat.calls[1] if m.role == "tool" and "limit reached" in m.content]
    assert len(refused) == 1


async def test_unknown_collection_creates_no_run(
    uow: UnitOfWorkFactory, store: InMemoryStore
) -> None:
    with pytest.raises(CollectionNotFoundError):
        await _agent(uow, FakeChatModel("x")).execute(AgentQuery(new_id(), "?"))
    assert store.agent_runs == {}


async def test_model_failure_marks_the_run_failed(
    uow: UnitOfWorkFactory, store: InMemoryStore, collection_id: UUID
) -> None:
    from app.application.ports.chat import ChatModelError

    with pytest.raises(ChatModelError):
        await _agent(uow, FakeChatModel(fail=True)).execute(AgentQuery(collection_id, "?"))
    [run] = store.agent_runs.values()
    assert run.status is RunStatus.FAILED
    assert "simulated outage" in (run.error or "")


class SlowChat(FakeChatModel):
    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[ChatStreamEvent]:
        await asyncio.sleep(1)
        async for event in super().stream(messages, tools):
            yield event


async def test_timeout_fails_the_run(
    uow: UnitOfWorkFactory, store: InMemoryStore, collection_id: UUID
) -> None:
    agent = _agent(uow, SlowChat("late"), timeout_seconds=0.05)
    with pytest.raises(AgentTimeoutError):
        await agent.execute(AgentQuery(collection_id, "?"))
    [run] = store.agent_runs.values()
    assert run.status is RunStatus.FAILED and "timed out" in (run.error or "")
