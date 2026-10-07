"""Search against real pgvector + full-text search, through the HTTP API."""

import asyncio
from typing import Any

import asyncpg
import pytest

from app.worker import run_worker
from tests.fakes import FakeReranker
from tests.integration.conftest import Api, PostgresUrls
from tests.integration.test_ingestion import GUIDE

pytestmark = pytest.mark.integration


async def _ingested_collection(api: Api, name: str, content: bytes = GUIDE) -> str:
    created = await api.client.post("/api/v1/collections", json={"name": name}, headers=api.headers)
    collection_id: str = created.json()["id"]
    upload = await api.client.post(
        f"/api/v1/collections/{collection_id}/documents",
        files={"file": ("networking.md", content, "text/markdown")},
        headers=api.headers,
    )
    assert upload.status_code == 202
    await run_worker(
        api.container, worker_id="t", poll_interval=0.01, stop=asyncio.Event(), once=True
    )
    return collection_id


async def _search(api: Api, collection_id: str, **body: Any) -> dict[str, Any]:
    response = await api.client.post(
        f"/api/v1/collections/{collection_id}/search", json=body, headers=api.headers
    )
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


def _paths(result: dict[str, Any]) -> list[list[str]]:
    return [hit["heading_path"] for hit in result["hits"]]


async def test_keyword_search_matches_any_stemmed_term(api: Api) -> None:
    collection_id = await _ingested_collection(api, "k8s")

    # A natural question: no chunk contains every word, and "routing" ≠ "routes"
    # literally. OR semantics + stemming still find the Ingress chunk first.
    result = await _search(
        api, collection_id, query="Which object is routing external traffic?", mode="keyword"
    )

    assert _paths(result)[0] == ["Ingress"]
    assert result["hits"][0]["keyword"]["rank"] == 1
    assert result["hits"][0]["vector"] is None


async def test_keyword_search_supports_explicit_search_syntax(api: Api) -> None:
    collection_id = await _ingested_collection(api, "k8s")

    phrase = await _search(api, collection_id, query='"stable virtual IP"', mode="keyword")
    excluded = await _search(api, collection_id, query="cluster -ClusterIP", mode="keyword")

    assert _paths(phrase) == [["Services"]]
    assert ["Services", "ClusterIP"] not in _paths(excluded)


async def test_vector_search_orders_by_similarity(
    api: Api, migrated_postgres: PostgresUrls
) -> None:
    collection_id = await _ingested_collection(api, "k8s")
    conn = await asyncpg.connect(migrated_postgres.app)
    try:
        target = await conn.fetchval(
            "SELECT contextual_text FROM chunks WHERE heading_path = '{Ingress}'"
        )
    finally:
        await conn.close()

    # The fake embedder maps equal texts to equal vectors: similarity 1.0.
    result = await _search(api, collection_id, query=target, mode="vector", top_k=3)

    assert _paths(result)[0] == ["Ingress"]
    assert result["hits"][0]["vector"]["score"] == pytest.approx(1.0, abs=1e-4)
    scores = [hit["score"] for hit in result["hits"]]
    assert scores == sorted(scores, reverse=True)


async def test_hybrid_scores_are_rrf_of_both_ranks(api: Api) -> None:
    collection_id = await _ingested_collection(api, "k8s")

    result = await _search(api, collection_id, query="ClusterIP default service type")

    # (The fake embedder's vectors carry no meaning, so only the RRF arithmetic and
    # the keyword side are asserted; the live demo shows real semantic behaviour.)
    assert result["mode"] == "hybrid"
    cluster_ip = next(h for h in result["hits"] if h["heading_path"] == ["Services", "ClusterIP"])
    assert cluster_ip["keyword"]["rank"] == 1
    for hit in result["hits"]:
        expected = sum(
            1 / (60 + retriever["rank"])
            for retriever in (hit["vector"], hit["keyword"])
            if retriever is not None
        )
        assert hit["score"] == pytest.approx(expected, abs=1e-6)
    scores = [hit["score"] for hit in result["hits"]]
    assert scores == sorted(scores, reverse=True)
    assert {"embed", "vector_search", "keyword_search", "total"} <= result["timings_ms"].keys()


async def test_search_never_leaks_other_collections(api: Api) -> None:
    first = await _ingested_collection(api, "first")
    await _ingested_collection(api, "second", GUIDE + b"\n## Extra\n\nOnly in second.\n")

    result = await _search(api, first, query="Only in second", mode="hybrid", top_k=10)

    assert ["Extra"] not in _paths(result)
    assert {hit["document_id"] for hit in result["hits"]} == {
        (
            await api.client.get(f"/api/v1/collections/{first}/documents", headers=api.headers)
        ).json()["items"][0]["id"]
    }


async def test_search_validation_and_not_found(api: Api) -> None:
    collection_id = await _ingested_collection(api, "k8s")

    bad = await api.client.post(
        f"/api/v1/collections/{collection_id}/search",
        json={"query": "x", "top_k": 20, "candidates": 5},
        headers=api.headers,
    )
    missing = await api.client.post(
        "/api/v1/collections/01a117c8-0000-7000-8000-000000000000/search",
        json={"query": "x"},
        headers=api.headers,
    )

    assert bad.status_code == 422
    assert missing.status_code == 404


async def test_reranked_search_through_the_api(api: Api) -> None:
    collection_id = await _ingested_collection(api, "k8s")
    api.container.reranker = FakeReranker(preferred={"Ingress"})

    reranked = await _search(api, collection_id, query="ClusterIP default service type")
    plain = await _search(api, collection_id, query="ClusterIP default service type", rerank=False)

    assert reranked["reranker"] == "fake"
    assert reranked["hits"][0]["heading_path"] == ["Ingress"]
    grades = [hit["rerank_score"] for hit in reranked["hits"]]
    assert grades == sorted(grades, reverse=True)
    assert all(isinstance(hit["retrieval_rank"], int) for hit in reranked["hits"])
    assert "rerank" in reranked["timings_ms"]
    assert plain["reranker"] is None
    assert all(hit["rerank_score"] is None for hit in plain["hits"])
