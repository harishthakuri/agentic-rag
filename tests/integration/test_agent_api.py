"""/agent/ask and /agent/runs through the HTTP API, over the real database."""

import json

import pytest

from app.application.ports.chat import ToolCall
from tests.fakes import FakeChatModel
from tests.integration.conftest import Api
from tests.integration.test_ask import _parse_sse
from tests.integration.test_search import _ingested_collection

pytestmark = pytest.mark.integration


def _search(query: str, call_id: str = "c1") -> list[ToolCall]:
    return [ToolCall(call_id, "search_knowledge_base", json.dumps({"query": query}))]


async def test_agent_run_as_json_and_replay(api: Api) -> None:
    collection_id = await _ingested_collection(api, "k8s")
    api.container.chat_model = FakeChatModel(  # type: ignore[assignment]
        [
            _search("external HTTPS traffic"),
            [ToolCall("c2", "read_more_context", '{"source": 1}')],
            "An Ingress routes it [1].",
        ]
    )

    response = await api.client.post(
        f"/api/v1/collections/{collection_id}/agent/ask",
        json={"question": "How does external traffic get in?"},
        headers=api.headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "An Ingress routes it [1]."
    assert body["cited"] == [1] and body["tool_calls"] == 2
    assert [s["tool"] for s in body["steps"]] == ["search_knowledge_base", "read_more_context"]
    assert any(source["cited"] for source in body["sources"])

    replay = await api.client.get(f"/api/v1/agent/runs/{body['run_id']}", headers=api.headers)
    run = replay.json()
    assert run["status"] == "succeeded"
    assert run["answer"] == body["answer"]
    assert [s["number"] for s in run["steps"]] == [1, 2]
    assert run["steps"][0]["arguments"] == {"query": "external HTTPS traffic"}
    # read_more_context ran the neighbours SQL: the source plus adjacent chunks.
    assert len(run["steps"][1]["result"]["sources"]) >= 2


async def test_agent_streams_steps_and_tokens(api: Api) -> None:
    collection_id = await _ingested_collection(api, "k8s")
    api.container.chat_model = FakeChatModel(  # type: ignore[assignment]
        [_search("ClusterIP"), "ClusterIP is internal [1]."]
    )

    response = await api.client.post(
        f"/api/v1/collections/{collection_id}/agent/ask",
        json={"question": "What is ClusterIP?", "stream": True},
        headers=api.headers,
    )

    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[:3] == ["run", "tool_call", "tool_result"]
    assert names[-1] == "done"
    assert set(names[3:-1]) == {"token"}
    assert events[1][1] == {
        "step": 1,
        "tool": "search_knowledge_base",
        "arguments": {"query": "ClusterIP"},
    }
    assert events[2][1]["result"]["sources"]
    assert events[-1][1]["run_id"] == events[0][1]["run_id"]


async def test_failed_run_is_recorded(api: Api) -> None:
    collection_id = await _ingested_collection(api, "k8s")
    api.container.chat_model = FakeChatModel(fail=True)  # type: ignore[assignment]

    response = await api.client.post(
        f"/api/v1/collections/{collection_id}/agent/ask",
        json={"question": "anything", "stream": True},
        headers=api.headers,
    )

    events = _parse_sse(response.text)
    assert events[0][0] == "run" and events[-1][0] == "error"
    run = await api.client.get(f"/api/v1/agent/runs/{events[0][1]['run_id']}", headers=api.headers)
    assert run.json()["status"] == "failed"
    assert "ChatModelError" in run.json()["error"]


async def test_unknown_run_is_404(api: Api) -> None:
    response = await api.client.get(
        "/api/v1/agent/runs/01a117c8-0000-7000-8000-000000000000", headers=api.headers
    )
    assert response.status_code == 404
