"""Infrastructure adapters that need no external services."""

import json
import math
from pathlib import Path

import httpx2 as httpx  # the OpenAI SDK is built on httpx2
import pytest
from openai import AsyncOpenAI

from app.application.ports.embeddings import EmbeddingUnavailableError
from app.domain.value_objects import EmbeddingSpec
from app.infrastructure.llm.openai_embedder import OpenAICompatibleEmbedder
from app.infrastructure.storage.local import LocalFileStorage


# --- Local file storage -------------------------------------------------------
async def test_storage_round_trip_and_delete_prefix(tmp_path: Path) -> None:
    storage = LocalFileStorage(tmp_path)
    await storage.save("c1/d1/a.md", b"hello")
    await storage.save("c1/d2/b.md", b"world")

    assert await storage.read("c1/d1/a.md") == b"hello"
    await storage.delete_prefix("c1/d1/")
    assert not (tmp_path / "c1" / "d1").exists()
    assert (tmp_path / "c1" / "d2" / "b.md").exists()


@pytest.mark.parametrize("key", ["../outside.txt", "c1/../../outside.txt", "/etc/passwd"])
async def test_storage_rejects_path_traversal(tmp_path: Path, key: str) -> None:
    storage = LocalFileStorage(tmp_path / "root")
    with pytest.raises(ValueError, match="escapes"):
        await storage.save(key, b"x")


async def test_storage_refuses_to_delete_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="root"):
        await LocalFileStorage(tmp_path).delete_prefix("")


# --- Embedder (against a mocked OpenAI-compatible HTTP API) -------------------
def _embedder(
    handler: httpx.MockTransport, *, dims: int = 3, batch: int = 2
) -> OpenAICompatibleEmbedder:
    client = AsyncOpenAI(
        base_url="http://ollama.test/v1",
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=handler),
    )
    return OpenAICompatibleEmbedder(client, EmbeddingSpec("m", dims), batch_size=batch)


def _recording_api(requests: list[dict[str, object]], vector: list[float]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        # Return items out of order: the adapter must sort by `index`.
        data = [
            {"object": "embedding", "index": i, "embedding": vector}
            for i in reversed(range(len(body["input"])))
        ]
        return httpx.Response(200, json={"object": "list", "data": data, "model": "m"})

    return httpx.MockTransport(handle)


async def test_documents_are_batched_and_vectors_normalised() -> None:
    requests: list[dict[str, object]] = []
    embedder = _embedder(_recording_api(requests, [3.0, 4.0, 0.0]))

    vectors = await embedder.embed_documents(["a", "b", "c"])

    assert [r["input"] for r in requests] == [["a", "b"], ["c"]]
    assert all(r["dimensions"] == 3 for r in requests)
    assert vectors[0] == pytest.approx([0.6, 0.8, 0.0])
    assert math.isclose(math.hypot(*vectors[0]), 1.0)


async def test_queries_get_the_instruction_prefix_documents_do_not() -> None:
    requests: list[dict[str, object]] = []
    embedder = _embedder(_recording_api(requests, [1.0, 0.0, 0.0]))

    await embedder.embed_query("what is a ClusterIP?")
    await embedder.embed_documents(["passage"])

    assert requests[0]["input"] == [
        "Instruct: Given a user question, retrieve passages from the knowledge base that "
        "answer it\nQuery: what is a ClusterIP?"
    ]
    assert requests[1]["input"] == ["passage"]


async def test_wrong_dimensions_are_rejected() -> None:
    embedder = _embedder(_recording_api([], [1.0, 0.0]), dims=3)
    with pytest.raises(ValueError, match="2 dimensions, expected 3"):
        await embedder.embed_documents(["a"])


async def test_server_errors_become_embedding_unavailable() -> None:
    embedder = _embedder(httpx.MockTransport(lambda request: httpx.Response(503)))
    with pytest.raises(EmbeddingUnavailableError):
        await embedder.embed_documents(["a"])
