"""backend/tests/security/test_csrf.py

Role: Prove the double-submit CSRF check covers every state-changing route, and that the three
exempt paths are exempt for the stated reason rather than by accident.

Why this project needs CSRF at all
----------------------------------
`.claude/rules/secure-coding-vapt.md` is explicit that the requirement is *conditional*:

    **CSRF**: Required *because* this project puts the JWT in an httpOnly cookie — the browser
    attaches it automatically, so cross-site requests would carry it. [...] If the token
    transport ever moves to an `Authorization` header, this requirement lapses — the two
    decisions are coupled, so change them together.

So the interesting failure is not "the middleware is missing" — it is **drift**: someone adds a
POST route six months from now and the check silently does not cover it, or someone adds a path
to `CSRF_EXEMPT_SUFFIXES` to fix a test. Both are caught by deriving the unsafe-route list from
the router itself (`_api_routes`) instead of hard-coding it, so a new POST is in scope the day
it ships.

The coupling is asserted too: `TestTheRequirementStillApplies` fails if the JWT stops living in
a cookie, because at that point this whole module is obsolete and should be deleted rather than
left quietly passing.

Why the token pair is generated, not fixed
------------------------------------------
`verify_csrf_token` compares the cookie against the header. A test that used the literal string
`"token"` for both would pass against an implementation that only checked *equality* — and an
implementation that only checks equality is exploitable, because an attacker who can set a
cookie (via a subdomain, or a response-splitting bug) controls both halves. So the tests below
use real `generate_csrf_token()` output and also assert that a structurally valid but
*unrelated* pair is rejected.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from backend.api.app import CSRF_EXEMPT_SUFFIXES, UNSAFE_METHODS, create_app
from backend.core.config import get_settings
from backend.core.security import generate_csrf_token, verify_csrf_token
from backend.models.user import User
from backend.tests.security.conftest import API, Stack, csrf_headers

Api = tuple[httpx.AsyncClient, Callable[[User | None], None]]


def _api_routes(app: FastAPI) -> list[APIRoute]:
    """Every `APIRoute`, reached through FastAPI 0.141's lazy router inclusion.

    `app.routes` does not contain the routes any more — an `include_router()` call leaves an
    `_IncludedRouter` wrapper holding `original_router`, so the obvious
    `[r for r in app.routes if isinstance(r, APIRoute)]` returns an empty list. That silently
    empty list is the reason `test_the_route_list_is_not_empty` exists: with no items, pytest
    turns each parametrized test into a single skip and the module reports green while
    asserting nothing about any route.

    `original_router` is private, and this walk is the only place that touches it. The
    alternative — deriving from `app.openapi()` — is public but only sees routes with
    `include_in_schema=True`, so a hidden route would escape the CSRF audit entirely. Covering
    every route matters more here than avoiding one private attribute, and
    `test_the_two_derivations_agree` fails loudly if the internal shape changes.
    """
    found: list[APIRoute] = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            found.append(route)
        nested = getattr(route, "original_router", None)
        if nested is not None:
            found.extend(_api_routes(nested))  # type: ignore[arg-type]
    return found


def _unsafe_routes() -> list[tuple[str, str]]:
    """Every (method, full path) that changes state.

    The router's own `route.path` is *un-prefixed* — the `/api/v1` prefix is applied at
    inclusion — so it is restored here. Getting that wrong would send every request to a path
    that 404s before the middleware's decision mattered.
    """
    app = create_app()
    routes = _api_routes(app)
    documented = set(app.openapi()["paths"])
    pairs: set[tuple[str, str]] = set()
    for route in routes:
        full = route.path if route.path in documented else f"{API}{route.path}"
        for method in route.methods & UNSAFE_METHODS:
            pairs.add((method, full))
    return sorted(pairs)


#: Derived rather than listed. A hand-written list is right the day it is written and wrong
#: on the day someone adds a route — which is exactly when this test needs to fire.
UNSAFE_ROUTES: list[tuple[str, str]] = _unsafe_routes()

#: Concrete bodies for the routes that need one, so a 422 from a missing body is not mistaken
#: for a CSRF pass. Keyed by path; anything absent is sent with no body.
BODIES: dict[str, dict[str, object]] = {
    # `question_text`, not `question` — the wrong name would 422 after the CSRF check and the
    # positive control would pass while never reaching the route it is meant to reach.
    f"{API}/queries": {"question_text": "What is the outbound domestic wire transfer fee?"},
    # `report` is required, and both enums are `StrEnum`s with lowercase values. The uppercase
    # member name 422s — which still satisfies "not a CSRF refusal" and so would have hidden
    # the fact that the positive control never reached the export route.
    f"{API}/analytics/export": {"report": "keywords", "format": "pdf"},
    f"{API}/auth/password": {
        "current_password": "DocQuery!Local2026",
        "new_password": "DocQuery!Local2027",
    },
    f"{API}/auth/login": {"email": "standard@tenant-a.example.com", "password": "x"},
    f"{API}/auth/register": {
        "email": "new@tenant-a.example.com",
        "password": "DocQuery!Local2026",
    },
}


def concrete(path: str, stack: Stack) -> str:
    """Substitute real ids for path parameters, so the request reaches the middleware.

    A `{document_id}` left literal would 422 on path validation *before* the CSRF middleware's
    decision mattered — and a 422 is not a 403, so the test would pass while proving nothing.
    """
    return (
        path.replace("{document_id}", str(stack.a.document.id))
        .replace("{query_id}", str(stack.a.query.id))
        .replace("{user_id}", str(stack.a.user.id))
    )


class TestTheRequirementStillApplies:
    """If the premise changes, this module should be deleted, not quietly kept green."""

    def test_the_jwt_is_still_carried_in_a_cookie(self) -> None:
        """The rule's condition, asserted as code.

        `secure-coding-vapt.md` ties the CSRF requirement to cookie transport. If a future
        change moves the token to an `Authorization` header, the requirement lapses — and this
        test failing is the signal to revisit the rule *and* this file together, rather than
        leaving a module that tests a control nobody needs any more.
        """
        settings = get_settings()
        assert settings.ACCESS_COOKIE_NAME, "no access cookie is configured"
        assert settings.CSRF_COOKIE_NAME and settings.CSRF_HEADER_NAME

    def test_the_cookie_is_samesite_strict(self) -> None:
        """`SameSite=Strict` is the other half of the control, not a replacement for it."""
        settings = get_settings()
        assert settings.COOKIE_SAMESITE.lower() == "strict", (
            f"COOKIE_SAMESITE is {settings.COOKIE_SAMESITE!r}; the rule requires Strict"
        )


class TestTheTokenComparison:
    """Unit-level, because these properties are cheap to assert and expensive to lose."""

    def test_a_matching_pair_verifies(self) -> None:
        token = generate_csrf_token()
        assert verify_csrf_token(token, token) is True

    def test_two_independently_generated_tokens_do_not_match(self) -> None:
        """Both halves are structurally valid; only their *equality* is missing."""
        assert verify_csrf_token(generate_csrf_token(), generate_csrf_token()) is False

    @pytest.mark.parametrize(
        ("cookie", "header"),
        [
            (None, None),
            ("", ""),
            (None, "abc"),
            ("abc", None),
            ("abc", ""),
            ("", "abc"),
        ],
    )
    def test_absent_or_empty_halves_never_verify(
        self, cookie: str | None, header: str | None
    ) -> None:
        """The empty-string case is the one that bites.

        `"" == ""` is true, so a comparison written as `cookie == header` accepts a request
        that carries neither — which is precisely the cross-site request this control exists to
        stop, since an attacker's page can send empty headers as easily as none.
        """
        assert verify_csrf_token(cookie, header) is False

    def test_a_token_is_long_enough_to_be_unguessable(self) -> None:
        token = generate_csrf_token()
        assert len(token) >= 32, f"a {len(token)}-character token is brute-forceable"
        assert token != generate_csrf_token(), "tokens are not unique per call"


class TestEveryUnsafeRouteIsCovered:
    """The drift guard. Parametrized over the router, so new routes join automatically."""

    def test_the_route_list_is_not_empty(self) -> None:
        """Guards the guard: an empty parametrization reports as green, not as a failure.

        This is not hypothetical. The first version of this module read `app.routes` directly,
        got `[]` because FastAPI 0.141 wraps included routers, and produced three skips and
        twenty-three passes — a module that audited zero routes and said so nowhere.
        """
        assert len(UNSAFE_ROUTES) >= 8, f"only found {len(UNSAFE_ROUTES)} unsafe routes"

    def test_the_two_derivations_agree(self) -> None:
        """The router walk and the OpenAPI schema must describe the same unsafe surface.

        If they diverge, either a route is hidden from the schema (and needs a deliberate
        decision) or `_api_routes` has stopped seeing routes because FastAPI's internals moved.
        Either way the audit above is no longer trustworthy, and this is the test that says so.
        """
        app = create_app()
        from_schema = {
            (method.upper(), path)
            for path, operations in app.openapi()["paths"].items()
            for method in operations
            if method.upper() in UNSAFE_METHODS
        }
        assert set(UNSAFE_ROUTES) == from_schema, (
            "the router walk and the OpenAPI schema disagree:\n"
            f"  only in the walk:   {sorted(set(UNSAFE_ROUTES) - from_schema)}\n"
            f"  only in the schema: {sorted(from_schema - set(UNSAFE_ROUTES))}"
        )

    @pytest.mark.parametrize(("method", "path"), UNSAFE_ROUTES)
    async def test_a_request_without_a_token_is_refused(
        self, stack: Stack, api: Api, method: str, path: str
    ) -> None:
        client, as_user = api
        as_user(stack.a.admin)
        url = concrete(path, stack)
        body = BODIES.get(path)

        response = await client.request(method, url, json=body)

        if any(path.endswith(suffix) for suffix in CSRF_EXEMPT_SUFFIXES):
            assert response.status_code != 403, (
                f"{method} {path} is on the exempt list but was refused as a CSRF failure"
            )
            return

        assert response.status_code == 403, (
            f"{method} {path} accepted a request with no CSRF token "
            f"({response.status_code}). Either the middleware does not cover it or it was "
            f"added to CSRF_EXEMPT_SUFFIXES."
        )
        assert response.json()["error"]["code"] == "csrf_failed", response.text

    @pytest.mark.parametrize(("method", "path"), UNSAFE_ROUTES)
    async def test_a_mismatched_pair_is_refused(
        self, stack: Stack, api: Api, method: str, path: str
    ) -> None:
        """Two valid tokens that are not each other. The attacker-with-a-cookie case."""
        if any(path.endswith(suffix) for suffix in CSRF_EXEMPT_SUFFIXES):
            pytest.skip("exempt by design; see CSRF_EXEMPT_SUFFIXES")

        client, as_user = api
        as_user(stack.a.admin)
        settings = get_settings()
        headers = {
            "Cookie": f"{settings.CSRF_COOKIE_NAME}={generate_csrf_token()}",
            settings.CSRF_HEADER_NAME: generate_csrf_token(),
        }

        response = await client.request(
            method, concrete(path, stack), json=BODIES.get(path), headers=headers
        )

        assert response.status_code == 403, (
            f"{method} {path} accepted a mismatched cookie/header pair"
        )

    @pytest.mark.parametrize(("method", "path"), UNSAFE_ROUTES)
    async def test_a_matching_pair_gets_past_the_middleware(
        self, stack: Stack, api: Api, method: str, path: str
    ) -> None:
        """The positive control, and the reason this module is not just `assert 403`.

        A middleware that rejected *everything* would satisfy every test above. What is
        asserted
        here is only that the response is not a CSRF refusal — the route may still 404, 422 or
        400 for its own reasons, and that is fine; those are decisions made after the check.
        """
        client, as_user = api
        as_user(stack.a.admin)

        response = await client.request(
            method, concrete(path, stack), json=BODIES.get(path), headers=csrf_headers()
        )

        # Not every success is JSON — `/analytics/export` returns a PDF or an XLSX body, and
        # calling `.json()` on it raises. A non-JSON response cannot be a CSRF refusal, since
        # the middleware's refusal is always the JSON error envelope.
        try:
            code = response.json().get("error", {}).get("code")
        except ValueError:
            code = None

        assert response.status_code != 403 or code != "csrf_failed", (
            f"{method} {path} rejected a matching token pair as a CSRF failure"
        )


class TestSafeMethodsAreNotBlocked:
    """A GET must not need a token — requiring one would break every page load."""

    @pytest.mark.parametrize(
        "path",
        [
            f"{API}/documents",
            f"{API}/history",
            f"{API}/audit",
            f"{API}/analytics/keywords",
            "/health",
        ],
    )
    async def test_a_get_needs_no_token(self, stack: Stack, api: Api, path: str) -> None:
        client, as_user = api
        as_user(stack.a.admin)
        response = await client.get(path)
        assert response.status_code == 200, f"{path} -> {response.status_code}"


class TestTheExemptionsAreNarrow:
    """Three suffixes are exempt. This asserts the list has not grown and cannot over-match."""

    def test_the_exempt_list_is_exactly_the_three_documented_paths(self) -> None:
        """A closed set, so adding a fourth is a deliberate act with a test to update.

        Each is exempt because there is no session cookie yet (`login`, `register`) or
        because a
        different control applies (`refresh`: `SameSite=Strict`, cookie path scope, and
        rotation). `logout` and `password` are deliberately *not* exempt.
        """
        assert set(CSRF_EXEMPT_SUFFIXES) == {
            "/auth/login",
            "/auth/register",
            "/auth/refresh",
        }

    @pytest.mark.parametrize("suffix", ["/auth/logout", "/auth/password", "/documents"])
    def test_the_state_changing_auth_routes_are_not_exempt(self, suffix: str) -> None:
        assert suffix not in CSRF_EXEMPT_SUFFIXES

    def test_a_path_that_merely_contains_an_exempt_suffix_is_not_exempt(self) -> None:
        """The match is on the *end* of the path, so a longer URL cannot borrow an exemption.

        A note on what is deliberately *not* asserted here. The obvious attack to try is a
        query
        string — `/api/v1/documents?next=/auth/login` — and `_is_csrf_exempt` does return True
        for that string. It is not a vulnerability, because the predicate is only ever called
        with `request.url.path`, and Starlette's `.path` excludes the query string; the value
        above cannot reach it. I wrote that assertion first, watched it fail, and traced it
        to my
        own wrong premise rather than to a bug.

        It is recorded here instead of deleted because the exemption's safety rests on that
        caller-side property. If `_is_csrf_exempt` is ever called with a raw target — a full
        request line, `str(request.url)`, or a proxy-supplied header — the query string becomes
        an exemption bypass on every state-changing route. See
        `test_the_predicate_is_only_called_with_a_bare_path`.
        """
        from backend.api.app import _is_csrf_exempt

        assert _is_csrf_exempt(f"{API}/auth/login") is True
        assert _is_csrf_exempt(f"{API}/auth/register") is True
        assert _is_csrf_exempt(f"{API}/auth/login/extra") is False
        assert _is_csrf_exempt(f"{API}/auth/loginx") is False
        assert _is_csrf_exempt(f"{API}/documents") is False
        assert _is_csrf_exempt(f"{API}/documents/auth/login-something") is False

    def test_the_predicate_is_only_called_with_a_bare_path(self) -> None:
        """The premise the exemption depends on, pinned in place.

        `request.url.path` carries no query string; `str(request.url)` does. A change from the
        first to the second would turn `?next=/auth/login` into a CSRF bypass for every unsafe
        route, and it would look like a harmless refactor in review.
        """
        import inspect

        from backend.api import app as app_module

        source = inspect.getsource(app_module)
        assert "_is_csrf_exempt(request.url.path)" in source, (
            "the CSRF exemption is no longer evaluated against `request.url.path`. If it now "
            "sees a query string or a full URL, `?next=/auth/login` exempts every route."
        )

    async def test_a_query_string_cannot_buy_an_exemption(
        self, stack: Stack, api: Api
    ) -> None:
        """The same premise, asserted end to end rather than by reading the source."""
        client, as_user = api
        as_user(stack.a.admin)

        response = await client.request(
            "DELETE", f"{API}/documents/{stack.a.document.id}?next=/auth/login"
        )

        assert response.status_code == 403, (
            f"a query string ending in an exempt suffix bypassed CSRF: {response.status_code}"
        )
        assert response.json()["error"]["code"] == "csrf_failed", response.text


class TestTheRefusalDoesNotLeak:
    async def test_the_csrf_error_says_nothing_about_the_expected_token(
        self, stack: Stack, api: Api
    ) -> None:
        """A helpful error here would be a working exploit.

        The check logs `has_cookie`/`has_header` booleans for the operator; the *client* is
        told
        only that the request failed.
        """
        client, as_user = api
        as_user(stack.a.admin)
        response = await client.request("DELETE", f"{API}/documents/{stack.a.document.id}")

        assert response.status_code == 403
        body = response.text
        for leak in ("expected", "cookie", "docquery_csrf", "X-CSRF-Token"):
            assert leak.lower() not in body.lower(), f"the refusal mentions {leak!r}: {body}"

    async def test_a_refused_request_changes_nothing(self, stack: Stack, api: Api) -> None:
        """The point of the control: the side effect must not have happened.

        Middleware ordering is what makes this true — CSRF runs before the route, so the
        service is never reached. Asserted rather than assumed, because a check moved into a
        dependency would still return 403 *after* the body had been parsed and, in some
        designs, after work had begun.
        """
        client, as_user = api
        as_user(stack.a.admin)
        before = stack.a.document.status

        response = await client.request("DELETE", f"{API}/documents/{stack.a.document.id}")

        assert response.status_code == 403
        after = await stack.documents.get(stack.a.document.id, stack.a.id)
        assert after is not None and after.status is before, (
            f"a CSRF-refused delete still changed the status to {after and after.status}"
        )
