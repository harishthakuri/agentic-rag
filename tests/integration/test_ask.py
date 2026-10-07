"""/ask through the HTTP API: JSON and Server-Sent Events, over the real database."""

import json
from typing import Any

import pytest

from tests.fakes import FakeChatModel
from tests.integration.conftest import Api
from tests.integration.test_search import _ingested_collection

pytestmark = pytest.mark.integration

QUESTION = "Which object routes external HTTPS traffic?"


def _parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


async def test_ask_returns_cited_answer_as_json(api: Api) -> None:
    collection_id = await _ingested_collection(api, "k8s")
    api.container.chat_model = FakeChatModel("An Ingress does that [1].")  # type: ignore[assignment]

    response = await api.client.post(
        f"/api/v1/collections/{collection_id}/ask",
        json={"question": QUESTION},
        headers=api.headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "An Ingress does that [1]."
    assert body["cited"] == [1] and body["invalid_citations"] == []
    assert body["sources"][0]["number"] == 1
    assert body["model"] == "fake-chat"
    assert body["usage"]["prompt_tokens"] == 100
    assert {"search", "generate", "total"} <= body["timings_ms"].keys()


async def test_ask_streams_server_sent_events(api: Api) -> None:
    collection_id = await _ingested_collection(api, "k8s")
    api.container.chat_model = FakeChatModel("Streaming works [1].")  # type: ignore[assignment]

    response = await api.client.post(
        f"/api/v1/collections/{collection_id}/ask",
        json={"question": QUESTION, "stream": True},
        headers=api.headers,
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "sources" and names[-1] == "done"
    assert set(names[1:-1]) == {"token"}
    assert "".join(data["text"] for name, data in events if name == "token") == (
        "Streaming works [1]."
    )
    assert events[-1][1]["cited"] == [1]


async def test_ask_unknown_collection_is_404_even_when_streaming(api: Api) -> None:
    response = await api.client.post(
        "/api/v1/collections/01a117c8-0000-7000-8000-000000000000/ask",
        json={"question": QUESTION, "stream": True},
        headers=api.headers,
    )
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"


async def test_chat_model_outage(api: Api) -> None:
    collection_id = await _ingested_collection(api, "k8s")
    api.container.chat_model = FakeChatModel(fail=True)  # type: ignore[assignment]
    url = f"/api/v1/collections/{collection_id}/ask"

    as_json = await api.client.post(url, json={"question": QUESTION}, headers=api.headers)
    streamed = await api.client.post(
        url, json={"question": QUESTION, "stream": True}, headers=api.headers
    )

    assert as_json.status_code == 503
    assert as_json.headers["retry-after"] == "10"
    # Headers were already sent when generation failed: the error arrives in-band.
    events = _parse_sse(streamed.text)
    assert events[0][0] == "sources"
    assert events[-1] == (
        "error",
        {"title": "Answer generation failed", "detail": "ChatModelError"},
    )
