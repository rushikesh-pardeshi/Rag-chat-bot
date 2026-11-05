"""backend/tests/test_auth_routes.py

Role: HTTP-level tests for the auth surface — CSRF, rate limiting, mass assignment, and the
401 paths.

**Why these run without a database.** Every control asserted here rejects the request
*before* any handler touches a session:

  * the CSRF check is middleware, so it answers 403 without resolving a single dependency;
  * `SlowAPIMiddleware` counts and rejects outside the handler entirely;
  * Pydantic validates the body before the service is constructed, so `extra='forbid'` is
    observable with nothing behind it;
  * `get_current_user` decodes the cookie first and raises on a missing, malformed, expired
    or wrong-type token before it ever calls the repository.

That ordering is the reason these gate items are verifiable on a laptop with no Postgres. It
also means these tests would catch a refactor that moved any of those checks *after* the
database call — which would be a real regression, because it would turn a cheap rejection
into a query an unauthenticated caller can trigger.

What is deliberately NOT here: anything needing a real user row. A *valid* token, a
successful login, and cross-tenant reads are covered at the service layer in
`test_auth_service.py`, and their HTTP equivalents are marked `requires_db`.
"""

from __future__ import annotations

import contextlib
import uuid
from datetime import timedelta

import httpx
import pytest

from backend.core.config import get_settings
from backend.core.limiter import limiter
from backend.core.security import (
    create_access_token,
    create_refresh_token,
    generate_csrf_token,
)

API = "/api/v1"


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Clear the limiter's counters around every test.

    The limiter is a module-level singleton on in-memory storage, so without this the tests
    would silently depend on execution order — the eleventh request in the *file* would 429,
    not the eleventh in the test.
    """
    limiter.reset()
    yield
    limiter.reset()


def _cookies(**pairs: str) -> dict[str, str]:
    """Build a `Cookie` request header.

    Sent as a header rather than through httpx's per-request `cookies=` argument, which is
    deprecated because its persistence semantics are ambiguous. Setting it on the shared
    client fixture instead would leak a session between tests, which is worse.
    """
    return {"Cookie": "; ".join(f"{name}={value}" for name, value in pairs.items())}


def _access(token: str) -> dict[str, str]:
    return _cookies(**{get_settings().ACCESS_COOKIE_NAME: token})


def _csrf_pair() -> tuple[dict[str, str], dict[str, str]]:
    """A matching (cookie header, CSRF header) pair that satisfies the double submit."""
    token = generate_csrf_token()
    settings = get_settings()
    return (
        _cookies(**{settings.CSRF_COOKIE_NAME: token}),
        {settings.CSRF_HEADER_NAME: token},
    )


def _auth_limit() -> int:
    """The attempt count from `AUTH_RATE_LIMIT` (e.g. 10 from "10/minute")."""
    return int(get_settings().AUTH_RATE_LIMIT.split("/")[0])


async def _burn_attempts(client: httpx.AsyncClient, count: int) -> None:
    """Spend `count` login attempts, ignoring how each one ends.

    Each attempt's outcome is deliberately not asserted. With no database reachable the
    handler raises rather than returning a 500 — Starlette's `ServerErrorMiddleware` logs
    the failure and re-raises, so the exception surfaces here instead of as a response.

    That does not weaken the test. `SlowAPIMiddleware` increments the counter *before* it
    calls the handler, so the attempt is charged either way — which is exactly the property
    the rate limit needs: a request that errors still counts, or an attacker could brute
    force for free by aiming at whatever is currently broken.
    """
    for _ in range(count):
        with contextlib.suppress(Exception):
            await client.post(
                f"{API}/auth/login",
                json={"email": "attacker@example.com", "password": "wrong-password-1"},
            )


class TestCSRF:
    """Double-submit enforcement on unsafe methods."""

    async def test_unsafe_request_without_a_csrf_token_is_rejected(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post(f"{API}/auth/logout")
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "csrf_failed"

    async def test_cookie_without_a_matching_header_is_rejected(
        self, client: httpx.AsyncClient
    ) -> None:
        """The cookie alone proves nothing — the browser sends it cross-site too."""
        cookie_header, _ = _csrf_pair()
        response = await client.post(f"{API}/auth/logout", headers=cookie_header)
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "csrf_failed"

    async def test_header_without_a_matching_cookie_is_rejected(
        self, client: httpx.AsyncClient
    ) -> None:
        _, headers = _csrf_pair()
        response = await client.post(f"{API}/auth/logout", headers=headers)
        assert response.status_code == 403

    async def test_mismatched_cookie_and_header_are_rejected(
        self, client: httpx.AsyncClient
    ) -> None:
        settings = get_settings()
        response = await client.post(
            f"{API}/auth/logout",
            headers={
                **_cookies(**{settings.CSRF_COOKIE_NAME: generate_csrf_token()}),
                settings.CSRF_HEADER_NAME: generate_csrf_token(),
            },
        )
        assert response.status_code == 403

    async def test_a_matching_pair_passes_the_csrf_check(
        self, client: httpx.AsyncClient
    ) -> None:
        """Asserts on the *code*, not the status.

        Logout with a valid CSRF pair gets past the middleware and then needs a database, so
        the status here depends on whether one is running. What must never appear is
        `csrf_failed` — that would mean a legitimate request was being blocked.
        """
        cookie_header, csrf_header = _csrf_pair()
        response = await client.post(
            f"{API}/auth/logout", headers={**cookie_header, **csrf_header}
        )
        body = response.json()
        assert body.get("error", {}).get("code") != "csrf_failed"

    @pytest.mark.parametrize("path", ["/auth/login", "/auth/register", "/auth/refresh"])
    async def test_the_pre_session_endpoints_are_exempt(
        self, client: httpx.AsyncClient, path: str
    ) -> None:
        """Login, register and refresh carry no session for a forged request to ride on.

        Requiring a token there would mean minting one for anonymous callers. The assertion
        is again on the code: these get past CSRF and fail later for other reasons.
        """
        response = await client.post(f"{API}{path}", json={})
        assert response.json().get("error", {}).get("code") != "csrf_failed"

    async def test_safe_methods_need_no_csrf_token(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get("/health")
        assert response.status_code == 200


class TestRateLimiting:
    """`AUTH_RATE_LIMIT` per IP on the credential endpoints."""

    async def test_the_eleventh_login_attempt_in_a_minute_is_rejected(
        self, client: httpx.AsyncClient
    ) -> None:
        """Brute-force protection that does not depend on the account existing.

        Attempt `limit + 1` never reaches the handler at all — the response comes from
        middleware, which is why it is a clean 429 here with nothing behind it.
        """
        await _burn_attempts(client, _auth_limit())

        blocked = await client.post(
            f"{API}/auth/login",
            json={"email": "attacker@example.com", "password": "wrong-password-1"},
        )
        assert blocked.status_code == 429
        assert blocked.json()["error"]["code"] == "rate_limited"

    async def test_the_429_carries_retry_after(self, client: httpx.AsyncClient) -> None:
        """Without `Retry-After` a client cannot back off, so it hammers and stays blocked.

        The login form reads this header to drive its countdown, so a regression here is
        visible as a button that re-enables too early and burns another attempt.
        """
        await _burn_attempts(client, _auth_limit())

        blocked = await client.post(
            f"{API}/auth/login",
            json={"email": "attacker@example.com", "password": "wrong-password-1"},
        )
        assert blocked.status_code == 429
        assert "retry-after" in {key.lower() for key in blocked.headers}
        assert int(blocked.headers["retry-after"]) > 0

    async def test_the_error_envelope_is_the_same_shape_as_every_other_failure(
        self, client: httpx.AsyncClient
    ) -> None:
        """slowapi's default body is a bare string; one client-side parser is the point."""
        await _burn_attempts(client, _auth_limit())

        blocked = await client.post(
            f"{API}/auth/login",
            json={"email": "attacker@example.com", "password": "wrong-password-1"},
        )
        body = blocked.json()
        assert set(body) == {"error"}
        assert {"code", "message"} <= set(body["error"])

    async def test_a_failing_attempt_still_counts_against_the_limit(
        self, client: httpx.AsyncClient
    ) -> None:
        """The counter increments before the handler runs.

        This is the property `_burn_attempts` relies on, and it matters on its own: if the
        limiter only charged successful requests, an attacker could guess for free against
        any endpoint that happens to be erroring.
        """
        payload = {"email": "attacker@example.com", "password": "wrong-password-1"}
        await _burn_attempts(client, _auth_limit() - 1)

        # One attempt still owed. It reaches the handler — where, with no database, it
        # raises. Either way it was not short-circuited, which is the observation.
        try:
            allowed = await client.post(f"{API}/auth/login", json=payload)
        except Exception:  # noqa: BLE001 - reaching the handler at all is the assertion
            reached_handler = True
        else:
            reached_handler = allowed.status_code != 429
        assert reached_handler, "the limiter rejected an attempt that was still within quota"

        blocked = await client.post(f"{API}/auth/login", json=payload)
        assert blocked.status_code == 429

    async def test_reading_a_profile_is_not_rate_limited(
        self, client: httpx.AsyncClient
    ) -> None:
        """`GET /auth/me` runs on every page load; limiting it would break the UI.

        It is safe to leave unlimited because it accepts no credentials — there is nothing
        to guess, so there is nothing to slow down.
        """
        for _ in range(15):
            response = await client.get(f"{API}/auth/me")
        assert response.status_code == 401


class TestMassAssignment:
    """`extra='forbid'`: privilege fields are rejected, not silently dropped."""

    async def test_a_client_supplied_role_is_rejected(
        self, client: httpx.AsyncClient
    ) -> None:
        """The whole request fails.

        Ignoring the extra field would also be safe today, but it would leave a request that
        *looks* like it asked for admin and got a 201 — and the day someone adds `role` to
        the model for an unrelated reason, silent-ignore becomes silent-privilege-escalation.
        """
        response = await client.post(
            f"{API}/auth/register",
            json={
                "email": "escalate@example.com",
                "password": "correct-horse-battery",
                "role": "ADMINISTRATOR",
            },
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"
        assert any(f["field"] == "role" for f in response.json()["error"]["fields"])

    async def test_a_client_supplied_tenant_id_is_rejected(
        self, client: httpx.AsyncClient
    ) -> None:
        """Naming your own tenant would be cross-tenant read access in one field."""
        response = await client.post(
            f"{API}/auth/register",
            json={
                "email": "escalate@example.com",
                "password": "correct-horse-battery",
                "tenant_id": str(uuid.uuid4()),
            },
        )
        assert response.status_code == 422
        assert any(
            f["field"] == "tenant_id" for f in response.json()["error"]["fields"]
        )

    async def test_a_client_supplied_is_active_is_rejected(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post(
            f"{API}/auth/register",
            json={
                "email": "escalate@example.com",
                "password": "correct-horse-battery",
                "is_active": True,
            },
        )
        assert response.status_code == 422

    async def test_a_short_password_is_rejected_before_any_work_happens(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post(
            f"{API}/auth/register", json={"email": "a@example.com", "password": "short"}
        )
        assert response.status_code == 422
        assert any(
            f["field"] == "password" for f in response.json()["error"]["fields"]
        )

    async def test_validation_errors_never_echo_the_submitted_value(
        self, client: httpx.AsyncClient
    ) -> None:
        """A 422 that repeats the body puts the attempted password in the client's logs."""
        secret = "hunter2-hunter2-hunter2"  # noqa: S105
        response = await client.post(
            f"{API}/auth/register", json={"email": "not-an-email", "password": secret}
        )
        assert response.status_code == 422
        assert secret not in response.text


class TestUnauthenticatedAccess:
    """`get_current_user` refuses every token it cannot fully trust."""

    async def test_no_cookie_is_a_401(self, client: httpx.AsyncClient) -> None:
        response = await client.get(f"{API}/auth/me")
        assert response.status_code == 401

    async def test_a_garbage_cookie_is_a_401_not_a_500(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get(f"{API}/auth/me", headers=_access("not-a-jwt"))
        assert response.status_code == 401

    async def test_an_expired_token_is_a_401(self, client: httpx.AsyncClient) -> None:
        expired = create_access_token(
            user_id=str(uuid.uuid4()),
            tenant_id=str(uuid.uuid4()),
            role="ADMINISTRATOR",
            expires_in=timedelta(minutes=-5),
        )
        response = await client.get(f"{API}/auth/me", headers=_access(expired))
        assert response.status_code == 401

    async def test_a_refresh_token_is_not_accepted_as_an_access_token(
        self, client: httpx.AsyncClient
    ) -> None:
        """Refresh tokens are long-lived and carry no role. Accepting one here would hand a
        stolen refresh token fifteen days of API access instead of one rotation."""
        token, _ = create_refresh_token(
            user_id=str(uuid.uuid4()), tenant_id=str(uuid.uuid4())
        )
        response = await client.get(f"{API}/auth/me", headers=_access(token))
        assert response.status_code == 401

    async def test_a_tampered_signature_is_a_401(
        self, client: httpx.AsyncClient
    ) -> None:
        token = create_access_token(
            user_id=str(uuid.uuid4()),
            tenant_id=str(uuid.uuid4()),
            role="STANDARD_USER",
        )
        header, payload, signature = token.split(".")
        forged = f"{header}.{payload}.{signature[:-4]}AAAA"
        response = await client.get(f"{API}/auth/me", headers=_access(forged))
        assert response.status_code == 401

    async def test_refresh_without_the_refresh_cookie_is_a_401(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post(f"{API}/auth/refresh")
        assert response.status_code == 401

    async def test_a_401_response_reveals_nothing_about_why(
        self, client: httpx.AsyncClient
    ) -> None:
        """Distinguishable messages let a caller probe which half of a token was wrong."""
        garbage = await client.get(f"{API}/auth/me", headers=_access("not-a-jwt"))
        wrong_type, _ = create_refresh_token(
            user_id=str(uuid.uuid4()), tenant_id=str(uuid.uuid4())
        )
        typed = await client.get(f"{API}/auth/me", headers=_access(wrong_type))
        assert garbage.json()["error"] == typed.json()["error"]


class TestResponseHardening:
    """Headers and body shape on the auth surface."""

    async def test_security_headers_are_present_on_an_auth_response(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get(f"{API}/auth/me")
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Referrer-Policy"] == "no-referrer"

    async def test_no_hsts_over_plain_http(self, client: httpx.AsyncClient) -> None:
        """Sending HSTS from a dev server pins localhost to https for a year."""
        response = await client.get(f"{API}/auth/me")
        if not get_settings().cookies_secure:
            assert "Strict-Transport-Security" not in response.headers

    async def test_every_failure_uses_the_same_envelope(
        self, client: httpx.AsyncClient
    ) -> None:
        """One shape across 401, 403 and 422 means the client needs one error parser."""
        unauthorized = await client.get(f"{API}/auth/me")
        forbidden = await client.post(f"{API}/auth/logout")
        unprocessable = await client.post(f"{API}/auth/register", json={})

        for response in (unauthorized, forbidden, unprocessable):
            body = response.json()
            assert set(body) == {"error"}, response.text
            assert {"code", "message"} <= set(body["error"])
