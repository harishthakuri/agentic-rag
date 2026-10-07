"""Browser UI: a single static page that talks to the public /api/v1 endpoints.

It has no server-side logic: the UI is just another API client, authenticated
with an API key like curl or a script. Served with a strict
Content-Security-Policy: only our own scripts and styles, no inline code, no
framing. Model output is untrusted and is always sanitised in the browser.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

STATIC_DIR = Path(__file__).parent / "static"

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",  # always revalidate: picks up new UI versions immediately
}


class _SecureStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers.update(SECURITY_HEADERS)
        return response


def mount_ui(app: FastAPI) -> None:
    app.mount("/ui", _SecureStaticFiles(directory=STATIC_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/ui/")
