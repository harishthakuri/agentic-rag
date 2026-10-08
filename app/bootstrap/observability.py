"""Process-wide observability: called once by each entry point (API, worker, evals).

The container gets the telemetry adapter for the use cases; this module sets up
the providers and exporters it reports to (see infrastructure/observability).
"""

from collections.abc import Callable
from typing import Any

from app.core.config import Settings
from app.infrastructure import observability


def configure_observability(settings: Settings, service_name: str) -> Callable[[], None]:
    """Returns a function that flushes and shuts down telemetry (call it at exit)."""
    return observability.configure(settings, service_name)


def instrument_fastapi(app: Any, settings: Settings) -> None:
    if settings.observability_enabled:
        observability.instrument_app(app)
