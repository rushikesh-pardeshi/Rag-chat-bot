"""backend/tests/test_error_cors.py

Role: Every error response must stay readable by a cross-origin browser — including a 500.

## The bug this exists for

Starlette installs the `@app.exception_handler(Exception)` handler on `ServerErrorMiddleware`,
which wraps the **whole** application. `CORSMiddleware` is registered as user middleware, so it
sits *inside* that wrapper. A 500 produced by the handler therefore never passes back out
through CORS and carries no `Access-Control-Allow-Origin`:

    401 -> access-control-allow-origin: http://localhost:3000
    500 -> access-control-allow-origin: None          <-- before the fix

A browser cannot read a cross-origin response without that header, so it throws the response
away and reports a **CORS policy error** instead of the 500. The consequence is not cosmetic:
the real failure becomes invisible in the browser, and the symptom accuses the one subsystem
that is working. A 500 from `/auth/refresh` was investigated as a missing preflight for
exactly this reason, and the actual cause — a foreign key violation on an audit row — was
nowhere on screen.

`surface_unhandled_errors` in `backend/api/app.py` fixes it by catching below CORS. These
tests pin that, because the property depends entirely on **middleware registration order** and
nothing about reordering `add_middleware` calls looks dangerous.

The assertions deliberately compare a 500 against a 401: "CORS works" is not the claim — the
claim is that CORS works *on the error paths too*, and the 401 is the control that proves the
test would notice a CORS misconfiguration rather than passing vacuously.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from backend.api.app import create_app
from backend.core.config import get_settings

#: An origin the app is configured to allow. Taken from settings rather than hardcoded, so
#: this does not quietly start testing the "disallowed origin" path if the default changes.
ALLOWED_ORIGIN = get_settings().cors_origins[0]

#: Mounted only in this module, to force the unhandled-exception path. Nothing in the real
#: surface is allowed to 500 on purpose, so there is no existing route to point at — and
#: `test_api_surface.py` pins the route list, so this must not be added to the app proper.
BOOM_PATH = "/__test_unhandled_exception"


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """The real app, plus one route that raises.

    `raise_app_exceptions=False` makes the transport behave like a server: it returns the 500
    response rather than re-raising into the test, which is the thing under test.
    """
    app = create_app()

    @app.get(BOOM_PATH)
    async def boom() -> None:
        raise RuntimeError("deliberate failure, to exercise the 500 path")

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as ac:
        yield ac


class TestErrorsStayReadableCrossOrigin:
    async def test_a_handled_401_carries_the_cors_header(
        self, client: httpx.AsyncClient
    ) -> None:
        """The control. Handled errors always worked; this proves the fixture is sound."""
        response = await client.get(
            "/api/v1/auth/me", headers={"Origin": ALLOWED_ORIGIN}
        )
        assert response.status_code == 401
        assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN

    async def test_an_unhandled_500_carries_the_cors_header(
        self, client: httpx.AsyncClient
    ) -> None:
        """The regression. Without `surface_unhandled_errors` this header is absent."""
        response = await client.get(BOOM_PATH, headers={"Origin": ALLOWED_ORIGIN})
        assert response.status_code == 500
        assert response.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN, (
            "a 500 lost its CORS headers, so a browser will report this as a CORS policy "
            "error and the real failure will be invisible. Check that the exception-trapping "
            "middleware is still registered *before* CORSMiddleware in _register_middleware."
        )

    async def test_the_500_body_is_the_generic_shape_and_leaks_no_detail(
        self, client: httpx.AsyncClient
    ) -> None:
        """The traceback belongs in the log, not in the response.

        An exception message can name an S3 bucket, a model arn or the contents of a row, so
        the body says nothing about the cause — `secure-coding-vapt.md`. Asserted because the
        middleware is a second place that builds this body, and the two must not drift.
        """
        response = await client.get(BOOM_PATH, headers={"Origin": ALLOWED_ORIGIN})
        assert response.json() == {
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred.",
            }
        }
        assert "deliberate failure" not in response.text
        assert "RuntimeError" not in response.text

    async def test_a_500_still_carries_the_request_id(
        self, client: httpx.AsyncClient
    ) -> None:
        """What a user can quote to get the traceback found.

        The id is attached by `request_context`, which is *inside* the trap, so a 500 that
        unwound past it would answer with no correlation id at all — leaving a support report
        with nothing to join on.
        """
        response = await client.get(BOOM_PATH, headers={"Origin": ALLOWED_ORIGIN})
        assert response.status_code == 500
        assert response.headers.get("x-request-id")

    async def test_a_disallowed_origin_gets_no_cors_header_on_a_500_either(
        self, client: httpx.AsyncClient
    ) -> None:
        """The fix must not have turned into a wildcard.

        Surfacing errors to the configured origins is the goal; surfacing them to *any* origin
        would be a change to the security boundary, and `allow_credentials=True` means the
        header is also what decides who may send the session cookie.
        """
        response = await client.get(
            BOOM_PATH, headers={"Origin": "http://attacker.example"}
        )
        assert response.status_code == 500
        assert "access-control-allow-origin" not in response.headers
