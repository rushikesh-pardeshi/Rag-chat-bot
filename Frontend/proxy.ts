/**
 * frontend/proxy.ts
 * Role: Optimistic auth redirect. Sends a visitor with no session cookie to /login, and a
 * visitor who already has one away from /login.
 *
 * **This is Next 16's `proxy.ts`, which replaced `middleware.ts`.** Same mechanism, new
 * name and new export. Do not re-add a `middleware.ts` — only one of the two is honoured.
 *
 * ## This is cosmetic. It is not access control.
 *
 * All it does is check whether a cookie is *present*. It does not verify the signature — the
 * JWT secret lives on the API and must not be shipped to the frontend — so a visitor who
 * hand-crafts a `docquery_access` cookie sails straight through. That is fine, because
 * nothing here grants access to data: every page fetches from the API, and the API validates
 * the token and enforces role plus `tenant_id` on every request. The only thing this saves
 * is a redirect flash for the overwhelmingly common case.
 *
 * Two consequences worth knowing:
 *
 *  1. **No role gate here.** Deciding "is this an Administrator?" would mean decoding an
 *     unverified token. Admin routes are gated client-side for rendering and server-side for
 *     real — see `hooks/useAuth.ts` and `require_admin` on the API.
 *  2. **Proxy runs on prefetches too.** A hovered `<Link>` triggers this, so it must stay
 *     free of network calls and database reads. Reading one cookie is the whole budget.
 *
 * **`ACCESS_TOKEN_EXPIRE_MINUTES` used to be the session length, and is not any more.** The
 * access cookie carries `max_age` equal to that setting, so the browser drops it on expiry;
 * this function once checked for that cookie alone and so redirected to /login on the next
 * navigation, even with a refresh cookie still valid for days. It cannot see the refresh
 * cookie — that one is scoped to `/api/v1/auth/refresh` and is never sent here.
 *
 * `SESSION_MARKER_COOKIE` below is the `Path=/` marker that fixes it, and the session now
 * lasts as long as the refresh token. Lowering `ACCESS_TOKEN_EXPIRE_MINUTES` shortens how
 * often a token is renewed, not how long somebody stays signed in.
 *
 * Cookie visibility: `docquery_access` is set by the API. Browsers scope cookies by host and
 * ignore the port, so in local development (`:8000` API, `:3000` UI, both `localhost`) the
 * proxy sees it. If the two are ever deployed on different hosts, set `COOKIE_DOMAIN` to the
 * shared parent domain — otherwise this check reads "no session" for everyone and the
 * redirect below has to be removed rather than left to misfire.
 */

import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

import { ROUTES } from '@/lib/constants';

/** Mirrors `ACCESS_COOKIE_NAME` in `backend/core/config.py`. */
const ACCESS_COOKIE = 'docquery_access';

/**
 * Mirrors `CSRF_COOKIE_NAME`. Read here as a *renewable session* marker, not for CSRF.
 *
 * `_set_session_cookies` writes all three cookies together, and gives this one `Path=/` and
 * the **refresh** token's lifetime (seven days) rather than the access token's five hours. So
 * its presence means "a session exists, or one that could still be renewed" — which is a
 * different and more useful question than the access cookie answers.
 *
 * This is the `Path=/` marker the note above asks for. Without it, a user whose access cookie
 * aged out mid-session was bounced to /login by this function even though their refresh cookie
 * had days left — the proxy cannot see the refresh cookie, which is scoped to
 * `/api/v1/auth/refresh`. Letting them through instead means the page loads, `useAuth` gets a
 * 401 from `/auth/me`, `api-client` renews silently, and they stay where they were going.
 *
 * Safe precisely because this function is not access control: see the note above. The worst a
 * forged marker buys is a page that renders its loading state and then redirects itself, having
 * been refused by the API.
 */
const SESSION_MARKER_COOKIE = 'docquery_csrf';

/**
 * Reachable without a session. Everything else under the matcher requires one.
 *
 * `/signup` has to be here for the obvious reason — a visitor creating an account has no
 * session yet — and `POST /auth/register` is CSRF-exempt on the API for the same reason
 * (`backend/api/app.py`, `CSRF_EXEMPT_SUFFIXES`). Registration is open to anyone who can reach
 * this origin and knows a tenant's email domain; see the F-01 note in `docs/RELEASE.md`.
 */
const PUBLIC_PATHS: readonly string[] = [ROUTES.home, ROUTES.login, ROUTES.signup];

function isPublic(pathname: string): boolean {
  return PUBLIC_PATHS.includes(pathname);
}

export function proxy(request: NextRequest) {
  const { pathname, search } = request.nextUrl;
  // Either cookie is enough. The access cookie means a live token; the marker means a session
  // that `api-client` can renew on the first 401. Requiring the first alone turned a renewable
  // session into a trip to the sign-in form every `ACCESS_TOKEN_EXPIRE_MINUTES`.
  const hasSession =
    request.cookies.has(ACCESS_COOKIE) || request.cookies.has(SESSION_MARKER_COOKIE);

  // Note what is *not* here: redirecting a signed-in visitor away from /login. The landing
  // route depends on the role, which is not knowable in the proxy, so `login/page.tsx` does
  // that redirect itself once `useAuth` has the server-verified user.
  if (hasSession || isPublic(pathname)) {
    return NextResponse.next();
  }

  // Preserve where they were headed so the form can return them there after signing in.
  // `login/page.tsx` accepts `next` only when it is a same-origin absolute path.
  const target = request.nextUrl.clone();
  target.pathname = ROUTES.login;
  target.search = '';
  target.searchParams.set('next', `${pathname}${search}`);
  return NextResponse.redirect(target);
}

export const config = {
  /**
   * Everything except Next internals, the metadata files, and anything with a file
   * extension. Static assets do not need a session check, and running on them would put
   * this function in the path of every image request.
   */
  matcher: ['/((?!_next/|favicon\\.ico|robots\\.txt|sitemap\\.xml|.*\\.[\\w]+$).*)'],
};
