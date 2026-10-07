from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.application.use_cases.system.check_readiness import CheckReadiness
from app.core.config import Settings
from app.main import create_app
from app.presentation.api.dependencies import get_check_readiness


class StubCheck:
    def __init__(self, name: str, healthy: bool) -> None:
        self.name = name
        self._healthy = healthy

    async def check(self) -> None:
        if not self._healthy:
            raise ConnectionError


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_live_returns_ok_and_request_id(client: TestClient) -> None:
    response = client.get("/health/live", headers={"X-Request-ID": "abc123"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"] == "abc123"


@pytest.mark.parametrize(("healthy", "expected_status"), [(True, 200), (False, 503)])
def test_ready_reflects_dependency_health(
    client: TestClient, healthy: bool, expected_status: int
) -> None:
    client.app.dependency_overrides[get_check_readiness] = lambda: CheckReadiness(  # type: ignore[attr-defined]
        [StubCheck("database", healthy)]
    )

    response = client.get("/health/ready")

    assert response.status_code == expected_status
    assert response.json()["dependencies"][0]["healthy"] is healthy
