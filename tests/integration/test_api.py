"""HTTP API against a real, migrated database, connected as the least-privilege role."""

import pytest

from app.domain.models import Document, DocumentType
from app.domain.value_objects import ContentHash
from tests.integration.conftest import Api

pytestmark = pytest.mark.integration


# --- Authentication -----------------------------------------------------------
async def test_requests_without_api_key_are_rejected(api: Api) -> None:
    response = await api.client.get("/api/v1/collections")

    assert response.status_code == 401
    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["title"] == "Unauthorized"


async def test_unknown_api_key_is_rejected(api: Api) -> None:
    response = await api.client.get(
        "/api/v1/collections", headers={"Authorization": "Bearer arag_wrong"}
    )
    assert response.status_code == 401


async def test_health_needs_no_api_key(api: Api) -> None:
    response = await api.client.get("/health/ready")
    database = next(d for d in response.json()["dependencies"] if d["name"] == "database")
    assert database["healthy"] is True


# --- Collections --------------------------------------------------------------
async def test_collection_crud(api: Api) -> None:
    created = await api.client.post(
        "/api/v1/collections",
        json={"name": "K8s-Docs", "description": "Kubernetes notes"},
        headers=api.headers,
    )
    assert created.status_code == 201
    body = created.json()
    assert body["name"] == "k8s-docs"
    assert body["embedding_model"] == "qwen3-embedding:8b"
    assert body["embedding_dim"] == 1024
    assert created.headers["location"].endswith(f"/api/v1/collections/{body['id']}")

    fetched = await api.client.get(f"/api/v1/collections/{body['id']}", headers=api.headers)
    assert fetched.json() == body

    listed = await api.client.get("/api/v1/collections", headers=api.headers)
    assert [c["id"] for c in listed.json()["items"]] == [body["id"]]

    deleted = await api.client.delete(f"/api/v1/collections/{body['id']}", headers=api.headers)
    assert deleted.status_code == 204
    missing = await api.client.get(f"/api/v1/collections/{body['id']}", headers=api.headers)
    assert missing.status_code == 404


async def test_duplicate_collection_name_conflicts(api: Api) -> None:
    await api.client.post("/api/v1/collections", json={"name": "docs"}, headers=api.headers)
    response = await api.client.post(
        "/api/v1/collections", json={"name": "docs"}, headers=api.headers
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "A collection named 'docs' already exists"


async def test_invalid_collection_name_is_a_problem_document(api: Api) -> None:
    response = await api.client.post(
        "/api/v1/collections", json={"name": "has spaces"}, headers=api.headers
    )

    assert response.status_code == 422
    assert response.json()["instance"] == "/api/v1/collections"


async def test_request_validation_errors_use_problem_details(api: Api) -> None:
    response = await api.client.post("/api/v1/collections", json={}, headers=api.headers)

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["errors"][0]["location"] == ["body", "name"]


# --- Documents (rows inserted directly until the upload endpoint exists) -------
async def test_documents_listing_and_cascade_delete(api: Api) -> None:
    created = await api.client.post(
        "/api/v1/collections", json={"name": "docs"}, headers=api.headers
    )
    collection_id = created.json()["id"]
    document = Document(
        collection_id=collection_id,
        title="Guide",
        source_filename="guide.md",
        document_type=DocumentType.MARKDOWN,
        content_hash=ContentHash.of(b"# Guide"),
        size_bytes=7,
        storage_key="docs/guide.md",
    )
    async with api.container.unit_of_work() as uow:
        await uow.documents.add(document)
        await uow.commit()

    listed = await api.client.get(
        f"/api/v1/collections/{collection_id}/documents", headers=api.headers
    )
    assert listed.status_code == 200
    [item] = listed.json()["items"]
    assert item["status"] == "pending"
    assert item["mime_type"] == "text/markdown"

    await api.client.delete(f"/api/v1/collections/{collection_id}", headers=api.headers)
    gone = await api.client.get(f"/api/v1/documents/{document.id}", headers=api.headers)
    assert gone.status_code == 404


async def test_duplicate_document_content_conflicts(api: Api) -> None:
    created = await api.client.post(
        "/api/v1/collections", json={"name": "docs"}, headers=api.headers
    )
    collection_id = created.json()["id"]

    def make() -> Document:
        return Document(
            collection_id=collection_id,
            title="Guide",
            source_filename="guide.md",
            document_type=DocumentType.MARKDOWN,
            content_hash=ContentHash.of(b"same bytes"),
            size_bytes=10,
            storage_key="docs/guide.md",
        )

    async with api.container.unit_of_work() as uow:
        await uow.documents.add(make())
        await uow.commit()

    from app.application.use_cases.documents import DuplicateDocumentError

    with pytest.raises(DuplicateDocumentError):
        async with api.container.unit_of_work() as uow:
            await uow.documents.add(make())
