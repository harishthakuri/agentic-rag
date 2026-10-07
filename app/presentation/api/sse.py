"""Server-Sent Events: a simple, HTTP-native way to stream to browsers and curl.

Each event is `event: <name>\\ndata: <json>\\n\\n`. Clients read the stream
with `EventSource` in a browser, or `curl -N`.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse
from pydantic import BaseModel


def sse_event(event: str, data: BaseModel | dict[str, Any]) -> str:
    payload = data.model_dump_json() if isinstance(data, BaseModel) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


def sse_response(events: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tell nginx-style proxies not to buffer the stream
        },
    )
