"""backend/tests/test_public_page_links.py

Role: Pin the rule that no publicly reachable page links to a session-gated route.

## The bug this exists to prevent coming back

`frontend/app/landing-cta.tsx` used to render an "Ask a question" button pointing at `/query`.
On a **production** build Next prefetches a `<Link>`, `frontend/proxy.ts` runs on prefetches,
and it answers a prefetch carrying no session cookie with a redirect to `/login`. App Router
then keeps that redirect in the client Router Cache.

The result was that signing in appeared to fail: `router.replace('/query')` replayed the cached
redirect and returned the user to the form they had just submitted with a valid session, and
the "Ask" tab in the signed-in nav kept bouncing to `/login` for the rest of the session.
Pressing refresh fixed it every time, because a document request skips the Router Cache —
which is what made it read as a flaky login rather than a caching bug.

`<Link>` prefetching is **off in `next dev` and on in a production build**, so this reproduced
on the POC instance (`WEB_MODE=prod`, `npm run start`) and never once locally. That asymmetry
is why it survived several rounds of looking at the sign-in code.

The fix was to stop linking to gated routes from public pages at all, rather than to disable
prefetching link by link — `prefetch={false}` on one link is a fix that the next person
adding a button does not know they have to repeat.

A source-text check, in the same style as `backend/tests/test_theme_default.py`, because this
repo has no JavaScript test runner: the only frontend gate installed is `tsc --noEmit`, and a
`<Link href>` to the wrong route is not a type error.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FRONTEND = REPO_ROOT / "frontend"

#: Pages an unauthenticated visitor can reach. Mirrors `PUBLIC_PATHS` in `frontend/proxy.ts`
#: plus the client islands those pages render, which is where the links actually live.
PUBLIC_SOURCES: tuple[str, ...] = (
    "app/page.tsx",
    "app/landing-cta.tsx",
    "app/login/page.tsx",
    "app/login/login-form.tsx",
    "app/signup/page.tsx",
)

#: Everything behind the proxy's session check. A `<Link>` to any of these from a file above is
#: the defect. Kept as the route literals *and* the `ROUTES` keys, because either spelling
#: reaches the same URL.
GATED_ROUTES: tuple[str, ...] = (
    "query",
    "documents",
    "history",
    "analytics",
    "audit",
    "users",
)


def _existing_public_sources() -> list[pathlib.Path]:
    paths = [FRONTEND / name for name in PUBLIC_SOURCES]
    present = [path for path in paths if path.exists()]
    assert present, f"none of the public sources exist under {FRONTEND}"
    return present


@pytest.mark.parametrize("source", _existing_public_sources(), ids=lambda path: path.name)
def test_public_page_does_not_link_to_a_gated_route(source: pathlib.Path) -> None:
    """No `<Link href>` or `router.push/replace` from a public page into a gated route.

    Matches both `href={ROUTES.query}` and `href="/query"`. The signed-in nav in
    `components/ui/page-shell.tsx` is where those entry points belong — it only ever renders
    once `useAuth` has a user, so its prefetches carry the session cookie.
    """
    text = source.read_text(encoding="utf-8")
    # Strip block and line comments: this file's own prose names the routes it forbids, and so
    # does `landing-cta.tsx`'s docstring explaining why they are absent.
    code = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"\{/\*.*?\*/\}", "", code, flags=re.DOTALL)

    offenders = [
        route
        for route in GATED_ROUTES
        if re.search(rf"\bROUTES\.{route}\b", code) or re.search(rf'["\']/{route}\b', code)
    ]
    assert not offenders, (
        f"{source.relative_to(REPO_ROOT)} links to session-gated route(s) {offenders}. "
        "A production build prefetches that link, proxy.ts answers the cookie-less prefetch "
        "with a redirect to /login, and App Router caches it — which breaks sign-in and the "
        "Ask tab until a hard refresh. See this module's docstring."
    )


def test_login_leaves_by_document_load_not_client_transition() -> None:
    """`handleSubmit` must use `window.location.assign`, never `router.replace`.

    A client transition after sign-in can replay a Router Cache entry written while the
    visitor was signed out. A document load rebuilds that cache and sends the cookie jar as it
    stands at request time, so it cannot.
    """
    form = FRONTEND / "app" / "login" / "login-form.tsx"
    assert form.exists(), f"missing {form}"
    text = form.read_text(encoding="utf-8")

    submit = text.split("async function handleSubmit", 1)
    assert len(submit) == 2, "handleSubmit not found in login-form.tsx"
    body = submit[1]

    assert "window.location.assign(target)" in body, (
        "the success path in handleSubmit must leave by a document load — see "
        "backend/tests/test_public_page_links.py for what a client transition replays"
    )
    assert "router.replace(target)" not in body, (
        "handleSubmit must not use router.replace: it can replay a redirect cached while the "
        "visitor was signed out, returning them to the form with a valid session"
    )


def test_proxy_still_runs_on_prefetches() -> None:
    """The premise of the rule above, pinned so it is noticed if it ever stops being true.

    If `proxy.ts` gains a prefetch opt-out (a `Next-Router-Prefetch` header check that returns
    `next()` early), the gated-link ban could be relaxed — and this assertion is where to find
    out that the trade-off changed.
    """
    proxy = FRONTEND / "proxy.ts"
    assert proxy.exists(), f"missing {proxy}"
    text = proxy.read_text(encoding="utf-8")
    assert "Next-Router-Prefetch" not in text, (
        "proxy.ts now inspects the prefetch header — revisit whether public pages may link to "
        "gated routes again, and update test_public_page_links.py's docstring"
    )
