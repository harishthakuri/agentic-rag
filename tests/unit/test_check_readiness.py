import asyncio

from app.application.use_cases.system.check_readiness import CheckReadiness


class FakeCheck:
    def __init__(self, name: str, *, error: Exception | None = None, delay: float = 0) -> None:
        self.name = name
        self._error = error
        self._delay = delay

    async def check(self) -> None:
        await asyncio.sleep(self._delay)
        if self._error:
            raise self._error


async def test_ready_when_all_checks_pass() -> None:
    report = await CheckReadiness([FakeCheck("db"), FakeCheck("llm")]).execute()

    assert report.ready
    assert [d.name for d in report.dependencies] == ["db", "llm"]


async def test_not_ready_when_a_check_fails_and_error_detail_is_not_leaked() -> None:
    failing = FakeCheck("db", error=ConnectionError("password=secret host=10.0.0.1"))

    report = await CheckReadiness([failing, FakeCheck("llm")]).execute()

    assert not report.ready
    db = report.dependencies[0]
    assert db.healthy is False
    assert db.error == "ConnectionError"


async def test_slow_check_times_out() -> None:
    report = await CheckReadiness([FakeCheck("db", delay=1)], timeout_seconds=0.01).execute()

    assert not report.ready
    assert report.dependencies[0].error == "timed out"
