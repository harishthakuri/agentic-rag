from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_root_redirects_to_ui(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


def test_ui_is_served_with_a_strict_content_security_policy(client: TestClient) -> None:
    response = client.get("/ui/")

    assert response.status_code == 200
    assert "<title>Agentic RAG</title>" in response.text
    csp = response.headers["content-security-policy"]
    assert "script-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "unsafe-inline" not in csp
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize(
    "asset",
    ["app.js", "styles.css", "vendor/marked-18.1.0.umd.js", "vendor/purify-3.4.16.min.js"],
)
def test_ui_assets_referenced_by_the_page_exist(client: TestClient, asset: str) -> None:
    assert asset in client.get("/ui/").text
    assert client.get(f"/ui/{asset}").status_code == 200


def test_ui_needs_no_api_key_but_the_api_does(client: TestClient) -> None:
    assert client.get("/ui/").status_code == 200
    assert client.get("/api/v1/collections").status_code == 401
